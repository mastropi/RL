# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:21:54 2022

@author: Daniel Mastropietro
@description: Definition of functions and classes used for the simulation of discrete-time MDPs with discrete actions, and DISCRETE or CONTINUOUS states.
When states are discrete, a 1D index is used to represent them.
When states are continuous, the state returned by the step() function performing a step of the MDP is the continuous-valued state. This continuous-valued
state is converted as needed to a 1D index by the getIndexFromState() method required to be defined in the environment with which the MDP agent interacts.
For instance, the continuous-valued state is converted to a 1D index when dealing with Fleming-Viot's absorption set and active set, as required for
e.g. the estimation of the occupation probability conditioned on non-absorption.
Note that this implies that every continuous-state environment must define a method to discretize the continuous-valued state which should be implemented
in the getIndexFromState() method. More details are given in the documentation for the Simulator class.

@notes: Note about the learning step counter, t: in all simulator processes (Monte-Carlo, Fleming-Viot, etc.) we use t to represent the time step counter
at which a Markov chain transitions to the NEXT state. Therefore t is initialize at 0 (as opposed to -1), so that after the first transition,
the time step is t = 1 (i.e. t signals precisely the time at which the particle is at the NEXT state, after the transition has taken place).
This follows the usual convention for continuous-time Markov processes where the event time is the time at which the Markov chain is at the NEXT state,
after the event takes place, i.e. the state to which the Markov chain transitions to.
See e.g. Pierre Bremaud's book, Chapter 13, graph of the jump rate process.
"""

import os
import sys
import copy
import warnings
import time
from datetime import datetime

from collections import deque   # Used for fast update of lists at the borders (which is a very common operation done on lists by the methods implemented here)
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
import gym

from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.episodic.discrete.fv import LeaFV
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
from Python.lib.estimators import DEBUG_ESTIMATORS
from Python.lib.estimators.fv import initialize_phi, estimate_stationary_probabilities, update_phi, update_phi_on_all_states
from Python.lib.agents.policies.parameterized import PolNN
from Python.lib.simulators.fv import reactivate_particle, StoppingCriterion
from Python.lib.simulators import DEBUG_TRAJECTORIES, MAX_NUMBER_OF_STEPS_FOR_EXPECTATION, MIN_NUM_CYCLES_FOR_EXPECTATIONS, choose_state_from_set, parse_simulation_parameters, show_messages

from Python.lib.utils.basic import find_signed_max_value, generate_datetime_string, get_current_datetime_as_string, is_integer, keep_dict_params_defined_in_function, measure_exec_time
from Python.lib.utils.computing import compute_expected_reward, compute_set_of_frequent_states_with_zero_reward, compute_survival_probability, mape, rmse
from Python.lib.utils.plotting import update_plots

#-- Constants used in the iterative plots
# Parameters for the windows containing the plots
WINDOW_TOP_LEFT_HORIZONTAL = 15
WINDOW_TOP_LEFT_VERTICAL = 35
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 480
SPACE_BETWEEN_WINDOWS = 15
# Parameters for the labels shown in some 2D plots (e.g. for the visit counts)
LABELS_FONTSIZE = 14
LABELS_COLOR = "orange"


class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a discrete-time, discrete-state, discrete-action environment

    Arguments:
    env: gym.Env
        Environment with discrete states and actions that is assumed to have the following methods, getters and setters defined:
        - seed(): to set the random generator seed
        - reset(): to reset the environment's state (see IMPORTANT note below)
        - isStateContinuous(): a method that returns whether the environment state is continuous.
        - getNumStates()
        - getNumActions()
        - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
            (responsible for defining the initial state when running the simulation)
        - getState() --> returns the current state of the environment (it can be a 1D index for discrete-state environments or anything for continuous-valued states such as e.g. the Mountain Car)
        - getIndexFromState() --> returns the 1D state index from a given environment state
        - getStateFromIndex() --> returns either the simulation state (used by the simulation process) or the actual physical environment state (used for informational purposes) from the 1D state index returned by getState()
        - getDimension()
        - getShape()
        - getAllStates()
        - getAllValidStates()
        - getTerminalStates()
        - getRewards()
        - getRewardsLandscape()
        - getInitialStateDistribution()
        - getV() --> returns the true state value function (which can be None, if unknown)
        - setInitialStateDistribution()
        - plot_values()
        - plot_points()
    IMPORTANT: The reset() method should return the index of the state to which the environment is reset.

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

        # Plotting handles that are used by different methods
        # Figure handle where the policy is plotted (if requested)
        self.fig_policy = None

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
        # Put the LEARNER of value functions in training mode when the value functions are approximated by a neural network (so far, done in the continuous state case)
        # This is important in case the neural network model has dropout layers, so that the dropout is actually used
        if self.env.isStateContinuous():
            self.agent.getLearner().getV().getModel().train()
            self.agent.getLearner().getQ().getModel().train()

        # Set the POLICY in evaluation mode (e.g. this is the time to compute the Critic of a policy, where the policy is evaluated, NOT trained)
        if isinstance(self.agent.getPolicy(), PolNN):
            self.agent.getPolicy().nn_model.eval()

        # Run the simulation and learning process
        if isinstance(self.agent.getLearner(), LeaFV):
            return self._run_fv(**kwargs)
        else:
            if self.agent.getLearner().getLearningTask() == LearningTask.CONTINUING:
                kwargs['nepisodes'] = 1
                return self._run_single_continuing_task(**kwargs)
            else:
                kwargs = keep_dict_params_defined_in_function(kwargs, self._run_single)
                return self._run_single(**kwargs)

    def _run_fv(self, t_learn=-1, max_time_steps=None,
                max_time_steps_for_absorbed_particles_check=+np.Inf, min_prop_absorbed_particles=1.0, stopping_criterion_fv=StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES,
                min_num_cycles_for_expectations=None,
                soft_killing=False,
                estimate_absorption_set=False, update_absorption_set_with_fv_visits=False, threshold_absorption_set=0.90,
                use_average_reward_stored_in_learner=False, use_fixed_average_reward=False, keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=True,
                reset_value_functions=True,
                epsilon_random_action=0.0,
                reward_for_exit_states: float=None,

                # DM-2025/01: Parameter that contains the policy learner used to update the policy after the initial exploration (using reward shaping) has finished
                # (so that reward shaping has actually an effect on the policy, which is the goal of doing reward shaping!)
                learner_policy=None,

                seed=None, verbose=True, verbose_period=100, plot=False, colormap="seismic", pause=0.1):
        """
        Runs all the simulations that are needed to learn differential value functions using the Fleming-Viot approach.

        These simulations include:
        - A simulation to estimate the absorption set A, if requested by parameter estimate_absorption_set.
        - A simulation on a single Markov chain used to estimate E(T_A), the expected reabsorption time to the set
        of uninteresting states, A, which is used as denominator of the Fleming-Viot estimator of stationary state probabilities.
        - A Fleming-Viot simulation on N particles which is used to estimate Phi(t, x) and P(T > t), which contribute
        to the numerator of the Fleming-Viot estimator of stationary state probabilities.

        Arguments:
        t_learn: (opt) int
            The learning step number (starting at 0) for which the FV simulation is run when FV is used in the context of FVAC,
            i.e. to learn an optimal policy using Fleming-Viot Actor-Critic.
            Negative values are accepted.
            This used for informational purposes, and to decide whether some learner resetting should be done, namely when t_learn <= 0,
            which indicates that a whole new simulation process (e.g. a new replication) is run,
            possibly using the same learner stored in the simulator object used for running the simulations.

        max_time_steps: (opt) int
            Maximum number of steps to run the simulation for, computed as the comprehensive number of transitions observed over ALL particles.
            This parameter is used to prevent the simulation from running for a VERY long time which can happen in some pathological cases
            (e.g. a deterministic or almost deterministic policy, already observed in the 2D labyrinth where the FV system may get stuck at the same
            configuration just because one of the start states is close to an obstacle and has an estimated policy that tells the agent to go
            most of the times in the direction of the obstacle! --even though that is not the optimal policy, it is estimated as such).
            This parameter is expected to take a value much larger than `max_time_steps_for_absorbed_particles_check` when this is set to a finite value.
            Set it to `np.Inf` in order to run the FV simulation until all N particles are absorbed at least once.
            default: None, in which case 100*N is used, where N is the number of particles in the FV system

        max_time_steps_for_absorbed_particles_check: (opt) int
            When stopping_criterion_fv=StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN,
            maximum number of steps to run the simulation before starting to check the proportion of particles absorbed at least once,
            using the threshold passed as `min_prop_absorbed_particles`.
            Set it to `np.Inf` in order to run the FV simulation until all the N particles are absorbed at least once or until `max_time_steps` steps
            have been taken.
            This parameter is expected to be smaller than `max_time_steps` and CANNOT be None.
            default: +np.Inf

        min_prop_absorbed_particles: (opt) float in [0, 1]
            Under the AVERAGE reward criterion, proportion of particles that should be absorbed at least once
            after the maximum number of steps has been reached, in order to avoid spending a very long time in the FV simulation,
            because in some cases it may not be easy for all particles to be absorbed.
            default: 1.0

        stopping_criterion_fv: (opt) StoppingCriterion
            See the documentation of the StoppingCriterion enum class for details.
            A few notes:
            - When the stopping criterion is MAX_TIME_STEPS or MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES,
            the average reward is computed AT THE END of the FV simulation, as opposed to iteratively, as is the case with the other two possible cases.
            - MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES: the check for the proportion of absorbed particles is done against the proportion given in
            parameter `min_prop_absorbed_particles`. If this minimum proportion is satisfied *regardless of how many number of overall steps have been taken*
            (a.k.a. simulation time), then the simulation stops.
            - MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN: this parameter value has an effect different
            from MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES only when the proportion given in parameter `min_prop_absorbed_particles` is less than 1.0.
            The condition of "enough time steps have been taken" is given by comparing the number of steps against the threshold given in parameter
            `max_time_steps_for_absorbed_particles_check`.
            This option may be useful for avoiding a too early stopping of the FV simulation should the survival times be rather large.
            deafult: StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES

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

        soft_killing: (opt) bool
            Whether to use FV particle system with soft killing, i.e. where each state has a probability of being killed when visited,
            as opposed to defining a fixed absorption set A.
            default: False

        estimate_absorption_set: (opt) bool
            Whether to estimate the absorption set from an initial exploration of the environment, based on the state visit frequency.
            default: False

        update_absorption_set_with_fv_visits: (opt) bool
            Whether to update the absorption set based on the visit frequency during the FV simulation.
            default: False

        threshold_absorption_set: (opt) float in [0, 1]
            Cumulative (when sorted from largest to smallest) proportion of state visit frequency below which a state is classified
            as part of the absorption set A.
            default: 0.90

        use_average_reward_stored_in_learner: (opt) bool
            In the AVERAGE reward criterion case, this indicates whether the average reward stored in the FV learner should be used as correction
            for the differential value functions estimation.
            When True, the average reward stored in the learner is used in all value function estimation instances,
            be it the single Markov chain excursion used to estimate E(T_A) or the FV simulation.
            When parameter `use_fixed_average_reward=True`, that average reward is iteratively updated by newly observed reward information,
            o.w. it is kept fixed throughout each simulation.
            Note that the average reward stored in the learner is updated after every simulation step, which includes:
            - the simulation run for the estimation of A (if requested by the user).
            - the initial simulation run to estimate E(T_A).
            - the FV simulation.

            Note that, for the iterative update of the average reward, we need to know the sample size behind the initial average reward stored in the learner,
            which weights the contribution of such initial average reward in the update formula used to compute the new average reward.
            This is possible because:
            - In the single Markov chain excursion, the GenericLearner class (from which the FV learner inherits) has an attribute
            (normally sample_size_initial_reward_stored_in_learner) that stores the sample size on which the average reward already stored
            in the learner is based upon (typically this sample size is updated at the end of the single Markov chain excursion with the newly observed sample size,
            which is exclusively based on the current learning process that just finished, meaning that the sample size does NOT increase w.r.t. the sample size
            behind the initial average reward passed when the current learning process started).
            - In the FV excursion, the initial average reward estimate receives a weight of N, the number of particles, as it is assumed to have come from
            a previous simulation carried out also on N particles, i.e. the same number of particles used in the current FV simulation and learning process.
            default: False

        use_fixed_average_reward: (opt) bool
            Whether to use a fixed average reward value in the correction of the value functions estimated by the learner during simulations.
            This condition applies to all simulations, be it the simulation run for the initial exploration of the environment or the FV simulation.
            The value of the average reward used in each case, is determined as follows:
            - for the initial exploration: using True in this parameter only has an effect when use_average_reward_stored_in_learner=True,
            in which case the average reward stored in the learner is used as fixed average reward value.
            Otherwise, the average reward is computed online, as the simulation proceeds.
            In a typical policy learning context, the average reward stored in the learner is the average reward estimated by the previous policy learning step,
            and of course that value exists so long as at least one policy learning step has already been run.
            - for the FV simulation: the average reward estimated by the initial exploration is used as fixed average reward value for value functions correction.
            Note that this is ALWAYS the case when the stopping criterion for FV (passed in dict_params_simul['stopping_criterion_fv']) requires to always
            reach MAX_TIME_STEPS, i.e. either when it is equal to MAX_TIME_STEPS or MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES.
            default: False

        keep_fv_estimation_of_average_reward_and_stationary_probability_consistent: (opt) bool
            Whether to guarantee that the average reward estimated by the FV estimator is consistent with the estimated stationary probability distribution
            outside A, meaning that the following equality is satisfied:
                "average reward estimated by FV" = \sum_{x outside A} p(x) r(x)
            Set this parameter to False if we are only interested in using the average reward signal for policy learning,
            as opposed to estimation purposes which require that the above consistency be satisfied.
            default: True

        reset_value_functions: (opt) bool
            Whether value function estimates, possibly stored in the value functions learner, should be reset at the beginning of the learning process
            (i.e. before launching the single Markov chain simulation used to estimate the expected reabsorption time, E(T_A)).
            Note that it might be useful to NOT reset the value functions if the value function learner is used to e.g. generate critic values used in
            an Actor-Critic policy learning context. In that case, we usually like to use the value function estimates obtained from the critic
            (i.e. from this value function learner) run when learning the policy at the PREVIOUS policy learning step (so that we don't start
            the learning of the value functions from scratch. This makes sense because the value functions associated to ONE policy learning step
            (of the Actor-Critic algorithm) are expected to be similar to those associated to the previous policy learning step.
            default: True

        epsilon_random_action: (opt) float in [0, 1]
            Probability to take a random action instead of choosing an action dictated by the policy.
            This is useful to guarantee ergodicity when policies are deterministic or have some actions with zero probability.
            default: 0.0

        Return: tuple
        Tuple with the following elements:
        - state_values: the estimated differential state value function V(s).
        - action_values: the estimated action value function Q(s,a).
        - state_counts: visit frequency of states for the whole simulation: E(T_A) excursion + FV simulation.
        - state_counts_et: visit frequency of states under the single Markov chain simulation that estimates E(T_A).
        - probas_stationary: dictionary with the estimated stationary probability for each state of interest.
        - expected_reward: the estimated expected reward.
        - n_events_a: number of steps during the estimation of the absorption set A (zero, if A is not estimated).
        - n_events_et: number of steps during the simulation of the single Markov chain used to estimate E(T_A) and P(T>t).
        - n_events_fv: number of steps during the FV simulation that estimates Phi(t).
        """
        if t_learn <= 0:
            # Reset the learner completely if this is the first step of a policy learning process (this is the default case, even if we are NOT in a policy learning process
            # but just running the simulation process just once, e.g. as part of a unit test, because in that case t_learn is expected to be set to its default value, -1)
            # Note that the first policy learning step (or the fact that t_learn <= 0, even if there is no policy learning process)
            # implies e.g. a start of a new replication, therefore no information currently stored from a previously executed replication should be present in the learner.
            # Note that in an FV learner, this may imply resetting collateral information to the information specific to the FV simulation, such as:
            # the absorption set, the activation set, the average reward observed during the initial exploration, the estimated expected reabsorption time, etc.
            self.agent.getLearner().reset(reset_episode=True, reset_value_functions=True, reset_average_reward=True, reset_auxiliary_info=True)
            print(f"[IN _run_fv() t_learn={t_learn}] The average reward stored in learner after RESET is: {self.agent.getLearner().average_reward}, {self.agent.getLearner()._average_reward_in_episode} (EPISODE)")
        # ALWAYS reset the number of steps run for the estimation of A and the initial exploration
        # Goals:
        # - Avoid having too many simulation steps run when no EXIT states are observed from A, thus giving a lot of budget to competing methods such as TDAC.
        # - Avoid having A grow too fast, perhaps...?
        #   (if a larger sample size is used to estimate A we might end up with more states in A which would complicate observing EXIT states...? not fully sure about this though)
        self.agent.getLearner().resetNumTimeStepsForExpectation()

        #--- Parse input parameters ---
        if min_num_cycles_for_expectations is None:
            if self.agent.getLearner().getLearningCriterion() == LearningCriterion.AVERAGE:
                min_num_cycles_for_expectations = MIN_NUM_CYCLES_FOR_EXPECTATIONS
            else:
                # The expected reabsorption cycle time E(T_A) does not need to be estimated for the DISCOUNTED reward criterion,
                # so we do not require a minimum number of cycles to be observed during the single Markov chain simulation.
                # This simulation is simply used to provide a first estimate of the value functions, but NOT of the expected reabsorption cycle time.
                min_num_cycles_for_expectations = 0
        dict_params_simul = dict({  'max_time_steps': max_time_steps,                         # Maximum number of time steps allowed for the N particles (comprehensively) in the FV simulation used to estimate the QSD Phi(t,x)
                                    'max_time_steps_for_absorbed_particles_check': max_time_steps_for_absorbed_particles_check,
                                    'min_prop_absorbed_particles': min_prop_absorbed_particles,
                                    'stopping_criterion_fv': stopping_criterion_fv,
                                    'N': self.agent.getLearner().getNumParticles(),
                                    'T': self.agent.getLearner().getNumTimeStepsForExpectation(),   # Maximum number of time steps allowed in each episode of the single Markov chain that estimates the expected reabsorption time E(T_A)
                                    'soft_killing': soft_killing,
                                    'proba_killing': dict(),
                                    'estimate_absorption_set': estimate_absorption_set,
                                    'update_absorption_set_with_fv_visits': update_absorption_set_with_fv_visits,
                                    'threshold_absorption_set': threshold_absorption_set,
                                    'absorption_set': self.agent.getLearner().getAbsorptionSet(),
                                    'activation_set': self.agent.getLearner().getActivationSet(),
                                    'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                                    'epsilon_random_action': epsilon_random_action,
                                    'reward_for_exit_states': reward_for_exit_states,
                                    'learner_policy': learner_policy,
                                    'seed': seed})
        dict_params_info = dict({'verbose': verbose,
                                 'verbose_period': verbose_period,
                                 't_learn': t_learn,
                                 'plot': plot,
                                 'colormap': colormap,
                                 'pause': pause}) # This indexes the learning epoch of the optimal policy
        #--- Parse input parameters ---

        # Create the particles as copies of the main environment
        envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(dict_params_simul['N'])]

        state_values, action_values, advantage_values, state_counts, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_absorption_cycles_used, \
            time_last_absorption, max_survival_time, n_events_a, n_events_et, n_events_fv = \
                self._estimate_value_functions_and_expected_reward_fv(  envs, dict_params_simul, dict_params_info,
                                                                        probas_stationary_start_state_et=self.agent.getLearner().getProbasStationaryStartStateET(),
                                                                        probas_stationary_start_state_fv=self.agent.getLearner().getProbasStationaryStartStateFV(),
                                                                        use_average_reward_stored_in_learner=use_average_reward_stored_in_learner,
                                                                        use_fixed_average_reward=use_fixed_average_reward,
                                                                        keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=keep_fv_estimation_of_average_reward_and_stationary_probability_consistent,
                                                                        reset_value_functions=reset_value_functions)

        return state_values, action_values, advantage_values, state_counts, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_absorption_cycles_used, n_events_a, n_events_et, n_events_fv

    def _estimate_value_functions_and_expected_reward_fv( self, envs, dict_params_simul, dict_params_info,
                                                          probas_stationary_start_state_et: dict=None,
                                                          probas_stationary_start_state_fv: dict=None,
                                                          use_average_reward_stored_in_learner=False,
                                                          use_fixed_average_reward=False,
                                                          keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=True,
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
            If the 'estimate_absorption_set' is True, the absorption set A will be estimated from an initial excursion of the Markov chain.

        dict_params_info: dict
            Dictionary containing information to display or parameters to deal with the information to display.
            Accepted keys are:
            - verbose: whether to be verbose during the simulation.
            - verbose_period: the number of simulation time steps at which to be verbose.
            - t_learn: the learning step number (starting at 0), when FV is used in the context of FVRL, i.e. to learn an optimal policy.
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.
            - plot: whether to generate an interactive plot showing the evolution of the state value function estimate.
            - colormap: name of the colormap to use for each new estimate of the state value function added to the plot (e.g. "seismic").
            - pause: number of seconds to pause after updating the plot with the new estimate of the state value function. MUST be > 0 if we want the
            plot to be updated!

        probas_stationary_start_state_et: (opt) dict
            Stationary distribution to use for the selection of the start state of the single Markov chain
            simulation used to estimate the expected reabsorption cycle time, E(T_A).
            Dictionary keys are the states and dictionary values are their probability of selection.
            Normally this is a uniform distribution on the states at the boundary of A.
            default: None, in which case the initial state distribution stored in the environment is used

        probas_stationary_start_state_fv: (opt) dict
            Stationary distribution to use for the selection of the start state of each FV particle used in the FV process.
            States are the dictionary keys and their probability of selection are the values.
            default: None, in which case the distribution is set by information collected during the initial exploration of the environment,
            typically to the EXIT state distribution (but other options are possible, depending on whether SOFT killing is used and on the visited states).

        use_average_reward_stored_in_learner: (opt) bool
            See description in _run_fv().
            default: False

        use_fixed_average_reward: (opt) bool
            See description in _run_fv().
            default: False

        keep_fv_estimation_of_average_reward_and_stationary_probability_consistent: (opt) bool
            See description in _run_fv().
            default: True

        reset_value_functions: (opt) bool
            See the description in _run_fv().
            default: True

        Return: tuple
        Tuple with the following elements:
        - state_values: the estimated differential state value function V(s).
        - action_values: the estimated action value function Q(s,a).
        - state_counts: state_counts_et + state_counts_fv = visit frequency of states for the whole simulation: E(T_A) excursion + FV simulation.
        - state_counts_et: visit frequency of states under the single Markov chain simulation that estimates E(T_A).
        - probas_stationary: dictionary with the estimated stationary probability for each state of interest.
        - expected_reward: the estimated expected reward.
        - expected_absorption_time: estimated expected absorption time E(T_A) used in the denominator of the FV estimator
        of the stationary state probabilities.
        - n_cycles: number of cycles observed to estimate E(T_A).
        - time_last_absorption: discrete time of the last absorption observed, used in the estimation of E(T_A).
        - max_survival_time: maximum survival time observed when estimating P(T>t).
        - n_events_a: number of steps during the estimation of the absorption set A (zero if A is not estimated).
        - n_events_et: number of steps during the simulation of the single Markov chain used to estimate E(T_A) and P(T>t).
        - n_events_fv: number of steps during the FV simulation that estimates Phi(t).
        """

        # -- Auxiliary functions
        is_estimation_of_denominator_unreliable = lambda: n_absorption_cycles_used == 0 or n_absorption_cycles_used < dict_params_simul['min_num_cycles_for_expectations']

        def parse_simulation_parameters_fv(dict_params_simul, dict_params_info, env):
            "Parses the simulation (and information) parameters for the FV simulation (including parameters for generic (non-FV) simulations)"
            dict_params_simul = parse_simulation_parameters(dict_params_simul, env)

            # Parse information parameters
            dict_params_info['t_learn'] = dict_params_info.get('t_learn', 0)
            dict_params_info['verbose'] = dict_params_info.get('verbose', False)
            dict_params_info['verbose_period'] = dict_params_info.get('verbose_period', 1)
            dict_params_info['plot'] = dict_params_info.get('plot', False)
            dict_params_info['colormap'] = dict_params_info.get('colormap', "coolwarm")
            dict_params_info['pause'] = dict_params_info.get('pause', 0.1)

            # Parse FV-specific simulation parameters
            dict_params_simul, less_frequently_visited_states_case, absorption_set_has_been_updated, n_events_a, average_reward_a = parse_absorption_parameters(dict_params_simul, dict_params_info)

            return dict_params_simul, dict_params_info, less_frequently_visited_states_case, absorption_set_has_been_updated, n_events_a, average_reward_a

        def parse_absorption_parameters(dict_params_simul, dict_params_info):
            """
            Computes the characteristics of the absorption dynamics to use for the FV estimation process

            The absorption dynamics can be either a soft killing or a hard killing, as follows:
            - In the soft killing case, each state has a probability of the process being killed when visiting the state.
            - In the hard killing case, an absorption set A is defined, containing all the states where the process is killed when visited.
            
            dict_params_simul: dict
                Dictionary containing the simulation parameters (e.g. 'seed', simulation time 'T', etc.).
            
            dict_params_info: dict
                Dictionary containing information parameters (e.g. 'verbose', etc.).

            Return: tuple
            Tuple with the following elements:
            - Updated dict_params_simul with updated values for the following keys:
                - 'proba_killing' when dict_params_simul['soft_killing'] is True, which is a dictinoary containing the killing probability for each state.
                - 'absorption_set' and 'activation_set' when dict_params_simul['soft_killing'] is False (or is not given)
                AND dict_params_simul['estimate_absorption_set'] is True.
            - less_frequently_visited_states_case: str, a description of the case that classifies the set of less frequently visited states
            by the initial exploration used to define the absorption set A. Either "N/A" when the absorption set is not estimated by the process
            (based on the visit frequency) or a number that identifies the case (e.g. 1, 2, 3) followed by a description of the case.
            - absorption_set_has_been_updated: whether the absorption set has been updated w.r.t. to the set previously stored in the learner.
            Even if the process is responsible for updating the absorption set, this may not be updated if it has become "sufficiently" large
            (based on dict_params_simul['max_prop_absorption_set'] which defaults to 0.80).
            - n_events_a: number of steps used during the estimation of the absorption set A (zero, if not estimated).
            - average_reward_a: the average reward observed during the estimation of the absorption set A (np.nan, if not estimated).
            """
            #----- Auxiliary functions -----
            def assertions_on_absorption_set(absorption_set):
                assert len(estimated_absorption_set) > 0, "The absorption set cannot be empty"
                # Check that the absorption set does not contain any states with non-zero reward
                _states_in_absorption_set_with_nonzero_reward = [s for s in estimated_absorption_set if
                                                                 self.env.getReward(self.env.getStateFromIndex(s, simulation=True)) != 0.0]
                assert len(_states_in_absorption_set_with_nonzero_reward) == 0, \
                    f"The absorption set must not contain states with non-zero reward. The following states in the absorption set have non-zero reward: " \
                    f"{_states_in_absorption_set_with_nonzero_reward}"
            #----- Auxiliary functions -----

            dict_params_simul['max_time_steps_for_absorbed_particles_check'] = dict_params_simul.get('max_time_steps_for_absorbed_particles_check', +np.Inf)
            dict_params_simul['min_prop_absorbed_particles'] = dict_params_simul.get('min_prop_absorbed_particles', 1.0)
            dict_params_simul['stopping_criterion_fv'] = dict_params_simul.get('stopping_criterion_fv', StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES)

            dict_params_simul['estimate_absorption_set'] = dict_params_simul.get('estimate_absorption_set', False)
            dict_params_simul['update_absorption_set_with_fv_visits'] = dict_params_simul.get('update_absorption_set_with_fv_visits', False)
            dict_params_simul['threshold_absorption_set'] = dict_params_simul.get('threshold_absorption_set', 0.90)
            dict_params_simul['max_prop_absorption_set'] = dict_params_simul.get('max_prop_absorption_set', 0.70)   #0.90 #0.70
            dict_params_simul['reward_for_exit_states'] = dict_params_simul.get('reward_for_exit_states', None)

            dict_params_simul['soft_killing'] = dict_params_simul.get('soft_killing', False)

            n_events_a = 0          # Number of steps during the estimation of the absorption set A (initialized to zero in case the absorption set is NOT estimated by this process)
            average_reward_a = 0.0  # Average reward during the estimation of the absorption set A (initialized to zero in case the absorption set is NOT estimated by this process, in order to avoid problems with missing average reward values)
            less_frequently_visited_states_case = "N/A"
            absorption_set_has_been_updated = False
                ## NOTE: The above information stored in `absorption_set_has_been_udpated` is used to confirm that reward shaping is in fact done when it is requested by the user,
                ## and to update the set of frequently visited states after update of the absorption set following the FV simulation,
                ## which are used as part of the strategy for defining the set of start states when no EXIT states from A are observed.
            if  dict_params_info['t_learn'] == 0 or \
                dict_params_simul['estimate_absorption_set'] or \
                dict_params_simul['soft_killing']:
                # Estimate the absorption set when:
                # a) it's the first learning step run (where we should start from scratch in terms of absorption set, think of a new replication as part of a set of experiments)
                # b) there is the directive of estimating the absorption set (e.g. to update it for a new learning step, if requested by the user)
                # c) when we are running the process under the SOFT KILLING mode, because the soft killing probabiligy is a function of the visit frequency, hence we should here
                # compute the visit frequency of states under a regular exploration of the environment.
                # Note that, in case (b), the updated absorption set is constructed as the union of the states in the newly estimated absorption set
                # and the absorption set already stored in the FV learner, as we want the absorption set only to grow, not shrink
                # (since all the states in any historic absorption set are states that have been frequently visited and have yielded no reward).
                # However, the growth of the absorption set is done as long as its proportion of all VALID environment states is smaller than the maximum allowed (e.g. 70%).
                # TODO: (2025/05/29) Think whether, when the absorption set cannot be grown more because it reached its maximum allowed proportion, we should set the absorption set to the NEWLY identified absorption set (which corresponds to the current policy, and may not include states that belonged to previously identified absorption sets, just because the policy changed since those were absorption sets were identified)
                # Continuing with the to-do task: I write this because the new absorption set responds to the CURRENT policy being applied, hence some states in historic absorption sets may be LESS frequently visited under the current policy...
                # Doing this shift in the states of the absorption set (instead of keeping the absorption set already stored in the learner) may allow EXIT states to be more likely observed...?
                # Not clear, because we would be removing states that are less likely... so exit from the new absorption set could be more difficult...?
                # The only advantage is that the shifted absorption set would correspond to more current information.
                # TODO: (2024/12/27) Think how to adapt this logic when the absorption set also contains states with non-zero reward... Do we need to remove those states from the absorption set so that they can be visited during the FV exploration in order to collect the rewards associated to them?
                # (continuing with the to-do task: on the other hand, if those states with non-zero reward become less visited by the updated policy, perhaps it means that we should not visit them because the optimal policy should not take the agent there...?)

                # Perform an initial exploration of the environment in order to define the absorption set based on visit frequency and no rewards
                # In this excursion, the start state is defined by the environment's initial state distribution
                print(f"\n**** ABSORPTION SET SELECTION ****")
                print(f"Estimating the absorption set based on cumulative relative visit frequency (<= {dict_params_simul['threshold_absorption_set']}) of states with NO reward from an initial exploration of the environment...")
                _average_reward_stored_in_learner_prev = self.agent.getLearner().getAverageReward()
                _learner, n_events_a, average_reward_a = self.run_exploration(t_learn=dict_params_info['t_learn'], max_time_steps=dict_params_simul['T'],
                                                                              epsilon_random_action=dict_params_simul['epsilon_random_action'],
                                                                              seed=dict_params_simul['seed'],
                                                                              verbose=dict_params_info['verbose'], verbose_period=dict_params_info['verbose_period'])

                # IMPORTANT STEP: The average reward observed during the exploration that estimates A is stored in the learner,
                # so that it can be used as initial (or definite, if `use_fixed_average_reward=True`) average reward for the initial exploration that estimates E(T_A).
                # This MAY be IMPORTANT because, if the average reward stored in the learner is the one computed by the FV simulation and estimator,
                # and this overestimates the average reward too much (because of the variability associated to the three different elements of the FV estimator),
                # then the learning of value functions by the Monte-Carlo simulation using a TD learner would probably be very UNSTABLE and make the policy learning process break
                # i.e. an UNLEARNING of the policy (of course, this was already observed in ~Jun-2025, o.w. I would not be writing this here).
                _average_reward_to_store =  (self.agent.getLearner().getSampleSizeForAverageReward() * _average_reward_stored_in_learner_prev + n_events_a * average_reward_a) / \
                                            (self.agent.getLearner().getSampleSizeForAverageReward() + n_events_a)  if use_average_reward_stored_in_learner \
                                                                                                                    else average_reward_a
                self.agent.getLearner().setAverageReward(_average_reward_to_store)
                # The sample size behind the average reward we store in the learner is ALWAYS JUST the sample size of the CURRENT exploration
                # despite the fact that we may have used the average reward stored in the learner when use_average_reward_stored_in_learner=True,
                # whose sample size should in principle contribute to the sample size of the average reward we are storing now...
                # However, the average reward stored in the learner is usually the average reward observed at a previous policy learning step,
                # which therefore could correspond to a DIFFERENT policy than the one we are using at the current policy learning step. Hence, we should "forget" about its value
                # after using it here, hence we also forget about the sample size behind its calculation.
                self.agent.getLearner().setSampleSizeForAverageReward(n_events_a)
                print(f"[ESTIMATION OF A using {dict_params_simul['T']} steps] average reward stored in learner = {self.agent.getLearner().getAverageReward()} (seed={dict_params_simul['seed']}, based on n={n_events_a} exploration steps + {use_average_reward_stored_in_learner and self.agent.getLearner().getSampleSizeForAverageReward() or 0} from average reward stored in learner)"
                      f"(compared to previous value of {_average_reward_stored_in_learner_prev} that could be OVERESTIMATED by FV)")
                print()

                # Compute the absorption set
                # Note: this process accepts environments with continuous states, thanks to calling _learner.getStateIndices() which always return a list of 1D state indices.
                _state_indices = _learner.getStateIndices()
                dist_state_counts = pd.Series(_state_indices).value_counts(normalize=True)  # Note: value_counts() returns the values sorted by decreasing frequency, nice! (unless we use argument sort=False)
                _set_of_frequent_states_with_zero_reward = compute_set_of_frequent_states_with_zero_reward(_state_indices, _learner.getRewards(), threshold=dict_params_simul['threshold_absorption_set'])
                # 2024/10/23: Use this option of cumulative=False to reproduce the result presented in the EWRL-2024 POSTER on the LABYRINTH results where the absorption set is defined on NON-CUMULATIVE relative frequency
                #_set_of_frequent_states_with_zero_reward = compute_set_of_frequent_states_with_zero_reward(_state_indices, _learner.getRewards(), threshold=dict_params_simul['threshold_absorption_set'], cumulative=False)
                print(f"Distribution of state frequency on n={_learner.getNumSteps()} steps:"
                      f"\n{pd.concat([pd.Series(dist_state_counts, name='f'), pd.Series(np.cumsum(dist_state_counts), name='F')], axis=1)}")

                if dict_params_info['t_learn'] == 0:
                    # Store the absorption set in the learner as this is the first learning step
                    # NOTE: Even if there is already an absorption set stored in the learner we should NOT consider it as a reference absorption set
                    # because the current execution of the process could correspond to a new replication run (on a different seed from the one used to store that absorption set).
                    # In fact, most likely an absorption set was created and stored in the FV learner during construction of the test class used to run this learning process,
                    # and this absorption set was most likely identified using a different seed than the one we used here.

                    # First check if the absorption set is empty and add at least one state to it if it is
                    if len(_set_of_frequent_states_with_zero_reward) == 0:
                        # Add at least one state to the absorption set, as it cannot be empty
                        # This state is chosen as one of the most common states according to the initial state distribution of the environment
                        _most_common_state_in_isd = np.argmax(self.env.getInitialStateDistribution())
                        estimated_absorption_set = set({_most_common_state_in_isd})
                        warning_msg = f"WARNING: The estimated absorption set is EMPTY. One of the most common states in the initial state distribution of the environment is defined as absorption set: {estimated_absorption_set}"
                        print(warning_msg)
                        warnings.warn(warning_msg)
                    else:
                        estimated_absorption_set = _set_of_frequent_states_with_zero_reward

                    # Store the absorption set in the learner
                    self.agent.getLearner().setAbsorptionSet(estimated_absorption_set)
                else:
                    # Read the absorption set stored in the learner and ADD any new states to it as this is not the first learning step and an absorption set is already stored
                    # We do this because we do NOT want to remove states already present in the absorption set because they were frequently visited under previous policies,
                    # so they proved to be uninteresting (because they had no reward). In addition, if those states are no longer part of the absorption set it means that the learned
                    # policy takes the agent away from them because they proved not to lead to non-zero reward states, so it doesn't harm to leave them in the absorption set,
                    # even if the policy changed and some of those states are no longer frequently visited.
                    # BUT MORE IMPORTANTLY, we do NOT want to fully update the absorption set A because we may end up in a situation that is not at all favorable for Fleming-Viot,
                    # namely one where the FV particles keep exploring uninteresting states because they are part of the active set! (this already happened and is the reason behind
                    # estimating the average reward by FV as ZERO as the policy becomes closer to optimal, because the reward is no longer observed due to the situation just described!)
                    _absorption_set_stored_in_learner = self.agent.getLearner().getAbsorptionSet()
                    estimated_absorption_set = _absorption_set_stored_in_learner.union(_set_of_frequent_states_with_zero_reward)
                    assert len(estimated_absorption_set) >= len(_absorption_set_stored_in_learner), \
                        f"The new absorption set must be equal or larger than the absorption set previously stored in the learner:" \
                        f"\nstored A = {_absorption_set_stored_in_learner} (n={len(_absorption_set_stored_in_learner)})" \
                        f"\nupdated A = {estimated_absorption_set} (n={len(estimated_absorption_set)})"

                    _absorption_set_increased = len(estimated_absorption_set) > len(_absorption_set_stored_in_learner)
                    if  dict_params_simul['soft_killing'] or \
                        _absorption_set_increased and dict_params_simul['T'] < MAX_NUMBER_OF_STEPS_FOR_EXPECTATION:
                        # If soft killing is used or o.w. if the number of steps T used in the initial exploration has NOT reached its maximum allowed,
                        # updated the absorption set A based on the recently visited states yielding no rewards.
                        # Reason to avoid increasing A when T has reached its maximum: this fact is an indicator that the absorption set has grown too much...
                        # ALSO: this condition also makes the increase decision of the absorption set INDEPENDENT OF THE AGENT KNOWING THE SIZE OF THE ENVIRONMENT,
                        # which is supposed to NOT know (i.e. the agent should not know anything about the environment apart from what it finds during exploration).

                        # Keep track of the size before update and update the absorption set to the new set just computed
                        _size_absorption_set_before_update = len(self.agent.getLearner().getAbsorptionSet())
                        absorption_set_has_been_updated, _number_of_new_states_in_absorption_set = update_absorption_set_if_not_too_large(estimated_absorption_set)

                        if not dict_params_simul['soft_killing'] and _number_of_new_states_in_absorption_set > 0:
                            # For the HARD killing case, increase the simulation time for the initial exploration by the increase in the absorption set
                            # Note that we do NOT increase T for the SOFT killing case because under SOFT killing we do NOT need to observe EXIT states
                            # in order to run the FV simulation.
                            _prop_increase_absorption_set = _number_of_new_states_in_absorption_set / _size_absorption_set_before_update
                            dict_params_simul = update_number_of_steps_for_expectation(dict_params_simul, increase_rate=_prop_increase_absorption_set,
                                                                                       reason=f"(due to increase of absorption set by {_prop_increase_absorption_set*100:.1f}%)")

                # Update the absorption and activation sets of the simulation parameters dictionary with the sets stored in the learner and possibly just updated
                assertions_on_absorption_set(estimated_absorption_set)
                dict_params_simul['absorption_set'] = self.agent.getLearner().getAbsorptionSet()
                dict_params_simul['activation_set'] = self.agent.getLearner().getActivationSet()

                # When SOFT killing is used, update the killing probability of the states based on the new state visit frequency, of which the probability is a function.
                # This is done to mimic the logic used in the HARD killing context where NO state is REMOVED from the absorption set --see justification above.
                if dict_params_simul['soft_killing']:
                    assert isinstance(dict_params_simul['proba_killing'], dict), "dict_params_simul['proba_killing'] must be defined and must be a dictionary"
                    assert dict_params_simul['absorption_set'].issubset(dist_state_counts.keys()), "The absorption set must be a subset of the states with positive killing probability"
                    for s in dist_state_counts.keys():
                        # Use this to define the killing probability as a linear function, i.e. equal to the visit frequency of the state
                        dict_params_simul['proba_killing'][s] = dist_state_counts[s]
                        # Use this to define the killing probability as a STEP function that is non-zero in the states in the absorption set A
                        #dict_params_simul['proba_killing'][s] = 0.50  # dict_params_simul['threshold_absorption_set']
                        # TODO: (2025/05/29) Implement a sigmoid function of the cumulative visit frequency proposed by Matt (computed on the *decreasing* visit frequencies) as killing probability, centered at e.g. F(x) = 0.90 (the larger this threshold, the larger the set of states which would potentially have a large killing probability --how large this probability is depends on the sigmoid parameter (which is an hyperparameter of the process)
                        # (continuing with the to-do task: this is the idea proposed by Matt which makes absolute sense and should solve the problems encountered with the other two strategies above --although I don't recall what they are.)

                # Store the state with smallest visit frequency during the initial exploration to use as start state for the E(T_A) simulation
                # whenever the probability distribution for the start state for the E(T_A) simulation has not yet been set (typically this is the case at the first learning step)
                # Goal: Start at a state that is most likely close to the boundary of A
                #dict_params_simul['start_states_for_et'] = set({dist_state_counts.index[-1]})  # THIS DOES NOT SEEM LIKE A GOOD IDEA... POLICY LEARNING FAILS!! (perhaps because it is only one point...?
                dict_params_simul['start_states_for_et'] = set(dist_state_counts.index).difference(dict_params_simul['absorption_set'])

                # ******
                # DM-2025/06/15: Disabled the following section because the information collected about the less frequently visited states is no longer used as a backup for
                # the selection of start states of the FV particles. I do not remove the section completely because there is a possibly useful step of removing
                # the less frequently visited states from the absorption set A in order to increase the probability of observing EXIT states from A and thus being able to
                # choose the start states for the FV particles (see the `_less_frequently_visited_states_to_remove_from_absorption_set` variable definition).
                # Note, however, that this removal process may at the same time regenerate holes in A which may have been filled by the high visit frequency observed
                # during the FV simulation, so it's not clear that it is a useful strategy.
                # ******
                # Store the set of states that were visited during the excursion but that are NOT part of the absorption set
                # This is useful if we want to choose the start states for the FV particle system among those states, which makes sense in the following situations:
                # - under soft killing, it would be possible to choose the FV start states OUTSIDE the set of frequently visited states, which is where we want to be during FV.
                # - under hard killing, it would be possible to choose the FV start states when the exit state probability could not be estimated from the initial exploration
                # Note that we look at the less frequently visited states and NOT just the EXIT states from A because potentially using these as candidates for the FV start states
                # at the next iteration amy most likely lead to states that are already in the next A set (thinking that our strategy of filling the holes in A consists of
                # adding the states that are frequently visited by the FV excursion which could also be states where the FV system gets stuck. So, if we use those states
                # again as start states in the upcoming policy iteration, they would most likely be states where the FV system will again get stuck...)
                if False:
                    dict_params_simul['backup_start_states_for_fv'] = set(dist_state_counts.index).difference(dict_params_simul['absorption_set'])
                    less_frequently_visited_states_case = "1 - GREAT, LESS FREQUENTLY VISITED STATES DURING INITIAL EXPLORATION are OUTSIDE A"
                    if len(dict_params_simul['backup_start_states_for_fv']) == 0:
                        print("WARNING: All visited states during the initial exploration are part of the absorption set."
                              " This would be a problem if we need to use them as BACKUP set for the start state of the FV particles, if no EXIT states from A are observed."
                              "\nTrying to solve this now...")
                        # Check if there are states stored in the set of less frequently visited states in the FV learner
                        print("1) Checking if the set of LESS frequently visited states during the FV simulation performed in the PREVIOUS learning step are OUTSIDE the absorption set just identified...")
                        _less_frequently_visited_states_not_in_absorption_set = self.agent.getLearner().getLessFrequentlyVisitedSet().difference(dict_params_simul['absorption_set'])
                        if len(_less_frequently_visited_states_not_in_absorption_set) > 0:
                            print(f"=== OK! ===: The set of visited states that are not in the absorption set is defined as the set of less frequently visited states during the last FV simulation that are NOT in the currently identified absorption set.")
                            dict_params_simul['backup_start_states_for_fv'] = _less_frequently_visited_states_not_in_absorption_set
                            less_frequently_visited_states_case = "2 - GOOD, THE LESS FREQUENTLY VISITED STATES BY FV IN PREVIOUS STEP are OUTSIDE A"
                        else:
                            # Add a few states to this set, namely the states with smallest visit frequency of the absorption set
                            # and REMOVE those states from the absorption set in the HARD killing setting
                            # (o.w. there would be a problem when selecting that state as starting state for the FV particles because the start state would belong to the absorption set
                            # and this is not allowed in the HARD killing setting)
                            _number_of_less_frequently_visited_states_to_remove_from_absorption_set = min(len(dist_state_counts), 5)
                            print(f"*** WARNING ***: Problem NOT solved: no visited state by the FV simulation at the PREVIOUS learning step is OUTSIDE the currently identified absorption set."
                                  f"\nThe set of visited states that are not in the absorption set is defined as the {_number_of_less_frequently_visited_states_to_remove_from_absorption_set} least frequently visited states in the absorption set.")
                            _less_frequently_visited_states_to_remove_from_absorption_set = set( sorted(dist_state_counts.index, key=lambda x: dist_state_counts[x])[:_number_of_less_frequently_visited_states_to_remove_from_absorption_set] )
                            if False and not dict_params_simul['soft_killing']:
                                # NOTE: (2025/05/23) I think I deactivated this step of removing states from A because it may lead to a decrease of the absorption set A, which is actually not allowed.
                                # HOWEVER, I think we could allow a decrease in A in order to accommodate this situation!
                                print(f"2) Removing those states from the absorption set: {sorted(_less_frequently_visited_states_to_remove_from_absorption_set)}")
                                dict_params_simul['absorption_set'] = dict_params_simul['absorption_set'].difference(_less_frequently_visited_states_to_remove_from_absorption_set)
                                self.agent.getLearner().setAbsorptionSet(dict_params_simul['absorption_set'])
                            dict_params_simul['backup_start_states_for_fv'] = _less_frequently_visited_states_to_remove_from_absorption_set
                            less_frequently_visited_states_case = "3 - BAD, THE LESS FREQUENTLY VISITED STATES are taken FROM A"
                    print(f"Less frequently visited states case: {less_frequently_visited_states_case}")

                print("\nAbsorption set (2D)")
                self.env.show_states(dict_params_simul['absorption_set'])
                print("Activation set (2D)")
                self.env.show_states(dict_params_simul['activation_set'])
                print("**** ABSORPTION SET SELECTION ****\n")

            return dict_params_simul, less_frequently_visited_states_case, absorption_set_has_been_updated, n_events_a, average_reward_a

        def update_number_of_steps_for_expectation(dict_params_simul, increase_rate=0.10, reason=""):
            """
            Increases the number of steps used for the initial exploration in an attempt to observe EXIT states

            The number of steps is stored in the FV learner so that it keeps being used in subsequent learning steps.
            """
            T_prev = dict_params_simul['T']
            dict_params_simul['T'] = int( min(dict_params_simul['T'] * (1 + increase_rate), MAX_NUMBER_OF_STEPS_FOR_EXPECTATION) )
            self.agent.getLearner().setNumTimeStepsForExpectation(dict_params_simul['T'])
            print(f"Parameter T increased from {T_prev} to {dict_params_simul['T']} ({(dict_params_simul['T'] / T_prev - 1) * 100:.1f}%) {reason}")

            return dict_params_simul

        def update_absorption_set_if_not_too_large(absorption_set, max_prop_absorption_set=0.90): #+np.Inf):
            """
            Updates the absorption set stored in the FV learner with the given `absorption_set` as long as it has not grown above
            the `max_prop_absorption_set` threshold computed out of the number of states known by the learner so far, and returns whether it has been updated.

            Note that the proportion of the absorption set is measured w.r.t. the number of VALID states in the environment,
            i.e. the states in the environment that CAN be visited (e.g. obstacles in labyrinths are not counted).
            This number of valid states is retrieved with the environment's method getValidStates().
            """
            _size_absorption_set = len(absorption_set)
            _n_valid_states = len(self.env.getAllValidStates())
            _n_known_states = max(_size_absorption_set, len(self.agent.getLearner().getKnownEnvironmentSet()))    # At least the agent knows the states in the absorption set just identified
            _prop_absorption_set = _size_absorption_set / _n_known_states
            if _prop_absorption_set >= max_prop_absorption_set:
                # The absorption set has become large enough, we won't update it
                print(f"[CHECK #2] Absorption set NOT updated, as it has become large enough: "
                      f"size = {_size_absorption_set} states ({_prop_absorption_set * 100}% of {_n_known_states} KNOWN states >= {max_prop_absorption_set * 100}%) ({_size_absorption_set / _n_valid_states * 100:.1f}% of EXISTING valid states)")
                absorption_set_has_been_updated = False
                number_of_new_states_in_absorption_set = 0
            else:
                # Set the absorption set in the learner, which also automatically updates the activation and active sets
                number_of_new_states_in_absorption_set = len(absorption_set) - len(self.agent.getLearner().getAbsorptionSet())
                assert number_of_new_states_in_absorption_set >= 0, f"The size of the absorption set must NOT decrease: number of new states = {number_of_new_states_in_absorption_set}"
                if absorption_set != self.agent.getLearner().getAbsorptionSet():
                    print(f"[CHECK #2] Absorption set UPDATED! (most likely increased) ({number_of_new_states_in_absorption_set} new states) --> "
                          f"New size = {_size_absorption_set} states ({_prop_absorption_set * 100:.1f}% of {_n_known_states} KNOWN states, {_size_absorption_set / _n_valid_states * 100:.1f}% of EXISTING valid states)")
                    self.agent.getLearner().setAbsorptionSet(absorption_set)
                    # DM-2025/06/26: The reset of the backup EXIT state to None has been disabled because now we keep the backup exit states that are OUTSIDE A
                    # when defining the start states for the FV simulation.
                    # Note that, since the absorption set A CANNOT be reduced, any state in the backup exit set that is NOT in A, must be part of the external boundary of A.
                    # The activation set should be reset (to None) because the activation set currently stored in the learner corresponds to a different absorption set
                    #self.agent.getLearner().setActivationSet(None)
                    #print(f"The activation set stored in the learner has been reset to None (because it may no longer be valid following the change in the absorption set).")
                    absorption_set_has_been_updated = True
                else:
                    absorption_set_has_been_updated = False

            return absorption_set_has_been_updated, number_of_new_states_in_absorption_set
        # -- Auxiliary functions

        # -- Parse input parameters
        # Set the simulation seed
        # Note: Even though the seed is set by the _run_single() method below (which receives a `seed` parameter)
        # we need to set the seed here for a reproducible selection of the start state of the simulation,
        # whenever the start state distribution on the absorption set is given by the user.
        np.random.seed(dict_params_simul['seed'])

        # Parse specific parameters, related to the simulation in general, and related to the absorption dynamics characteristics
        dict_params_simul, dict_params_info, less_frequently_visited_states_case, absorption_set_has_been_updated, n_events_a, average_reward_a = \
            parse_simulation_parameters_fv(dict_params_simul, dict_params_info, envs[0])

        estimated_average_reward_before_initial_exploration = self.agent.getLearner().getAverageReward() if use_average_reward_stored_in_learner \
                                                                                                         else average_reward_a
        print(f"[IN _estimate_value_functions_and_expected_reward_fv()] The average reward before initial exploration is: {estimated_average_reward_before_initial_exploration}")
        # -- Parse input parameters

        # -- Step 1: Simulate a single Markov chain to estimate the EXIT state distribution from A and the expected cycle time of return to A, E(T_A)
        # Simulate the single Markov chain until EXIT states are observed from A or, if soft killing is used, only simulate it ONCE (as EXIT states are NOT used in soft killing)
        n_events_et = 0
        n_initial_explorations_run = 0
        flag_soft_killing_or_exit_states_observed_or_maximum_t_reached = False
        while not flag_soft_killing_or_exit_states_observed_or_maximum_t_reached:
            # We now define the start state for the VERY FIRST episode.
            # (At subsequent episodes, the start state is defined by the initial state distribution (isd) of the environment,
            # because that is the distribution that allows converting the originally EPISODIC task into a CONTINUING task.)
            # For the start state at the very first episode, since the estimation of E(T_A) requires full entrance cycles to A,
            # it is better to start the simulation OUTSIDE A (so that an entrance event to A will be observed with high probability),
            # and in particular following the stationary EXIT distribution, as required by the theory.
            # An estimate of this stationary EXIT distribution is normally stored in input parameter probas_stationary_start_state_et,
            # following its definition during the PREVIOUS policy learning step (where it was set equal to probas_stationary_start_state_fv).
            # Note that such probas_stationary_start_state_et distribution may actually NOT give a distribution of the EXIT states from A
            # because the set A may well have been updated just above! (when estimate_absorption_set = True)
            # Nevertheless, this is NOT a functional problem; the only problem is that the agent may take more time to observe the entrance event to A
            # for the first time, because the start state is part of a set of frequently observed states, which is how A is defined, thus delaying
            # the time when samples used to estimate E(T_A) can be taken.
            # When probas_stationary_start_state_et is None (which is the case at the very first step of a policy learning process),
            # the start state is chosen uniformly as the set of states defined after the initial exploration of the environment in entry dict_params_simul['start_states_for_et']
            # as long as it is not empty (it should never be empty though, because of the way it is defined above).
            # Otherwise, if for some reason it happens to be empty, we set the start state to None so that the state is chosen
            # following the initial state distribution of the environment (isd) (done by the env.reset() when called in the _run_single_continuing_task() method).
            if probas_stationary_start_state_et is None or len(probas_stationary_start_state_et) == 0:
                # DM-2025/06/14: We should NOT use information about the activation set because that would be cheating,
                # as the agent is assumed not to know anything about the environment, except for what it discovers while exploring.
                # if self.agent.getLearner().getActivationSet() is not None and len(self.agent.getLearner().getActivationSet()) > 0:
                #    start_state = choose_state_from_set(self.agent.getLearner().getActivationSet(), None)
                # else:

                # Note: setting the start_state value to `None` (when the 'start_states_for_et' entry in dict_params_simul is empty)
                # means that the start state will be chosen from the environment's initial state distribution
                # (done by _run_single_continuing_task() when choosing the start state of the simulation)
                start_state = choose_state_from_set(dict_params_simul['start_states_for_et']) if len(dict_params_simul.get('start_states_for_et', {})) > 0 else None
                print(f"The start state for the E(T_A) simulation has been chosen UNIFORMLY at random from the following set:")
                self.env.show_states(dict_params_simul.get('start_states_for_et'))
            else:
                start_state = choose_state_from_set(set(probas_stationary_start_state_et.keys()), probas_stationary_start_state_et)
                print(f"The start state for the E(T_A) simulation has been chosen as a random sample from the following distribution:"
                      f"\n{probas_stationary_start_state_et}"
                      f"\ntaking values on the following set,", end=" ")
                self.env.show_states(set(probas_stationary_start_state_et.keys()))
            print(f"SINGLE simulation on T={dict_params_simul['T']} steps for the estimation of the expected reabsorption time E(T_A) starts at state s={start_state}")
            if start_state is None:
                warning_msg = "WARNING: No information is available for choosing the start state of the E(T_A) simulation (=> start_state = None). " \
                              f"The start state will be chosen using the initial state distribution of the environment:" \
                              f"\n{self.env.getInitialStateDistribution() if len(self.env.getInitialStateDistribution()) <= 20 else 'Not printed because too large (' + str(len(self.env.getInitialStateDistribution())) + ' elements)'}"
                print(warning_msg)
                warnings.warn(warning_msg)
            time.sleep(0.5)

            state_values, action_values, advantage_values, state_counts_et, _, _, learning_info = \
                self._run_single_continuing_task(
                                t_learn=dict_params_info['t_learn'],
                                max_time_steps=dict_params_simul['T'],      # Max simulation time over ALL episodes
                                start_state_first_episode=start_state,
                                #estimated_average_reward=0.0,  # (2025/05/18) Use this (TOGETHER WITH use_fixed_average_reward=True) for the ABLATION study of estimating the average reward at a wrong value (e.g. always 0)
                                # NOTES: (2025/07/14)
                                # 1) The value of parameter estimated_average_reward is kept always fixed at estimated_average_reward_before_initial_exploration,
                                #   as opposed to being updated using the average reward observed at every simulation run in this loop as contribution,
                                #   because, if no EXIT states from A are observed, then NO reward is observed during that simulation
                                #   (as A only contains states with no reward, by definition) and we don't want to diminish the value of the average reward used as initial average.
                                #   Nevertheless, if no EXIT states are observed and the absorption set A was estimated at this policy learning step then, most likely,
                                #   the value of estimated_average_reward_before_initial_exploration is ALSO zero, because its value would be equal to the average reward observed
                                #   during the A-estimation process.
                                # 2) There is a conceptual difference between the average reward estimated by this INI exploration and the average reward estimated by the
                                #   A-estimation exploration run above: the start state.
                                #   - The start state in the A-estimation exploration is an environment's start state (e.g. the START cell in a labyrinth).
                                #   - The start state in this INI exploration is a state in the boundary of A (or close enough, based on the caveat mentioned in the comment above
                                #   about the states in the start state distribution stored in probas_stationary_start_state_et.
                                #   HENCE, the average reward of this INI exploration may be slightly biased upward because it is likely that a reward is observed at the beginning
                                #   of the exploration, if the policy at the INI start state is close to optimal, whereas the policy at the S state is far from optimal.
                                #   (this was observed already in a 6x8 random labyrinth (seed=4217) with WIND=0.6 , that's why I am writing this!)
                                #   In any case, I don't think this small bias would have a large impact in a bad learning of the value functions...but haven't really checked that.
                                estimated_average_reward=estimated_average_reward_before_initial_exploration,
                                use_fixed_average_reward=use_fixed_average_reward,
                                reset_value_functions=reset_value_functions,
                                epsilon_random_action=dict_params_simul['epsilon_random_action'],
                                reward_shaping=dict_params_simul['reward_for_exit_states'] is not None,
                                reward_for_exit_states=dict_params_simul['reward_for_exit_states'] if absorption_set_has_been_updated else np.nan,
                                seed=dict_params_simul['seed'] + n_initial_explorations_run,
                                set_cycle=dict_params_simul['absorption_set'] if not dict_params_simul['soft_killing'] else None,
                                dict_proba_cycle=dict_params_simul['proba_killing'] if dict_params_simul['soft_killing'] else None,
                                plot=dict_params_info['plot'],
                                verbose=dict_params_info['verbose'],
                                verbose_period=dict_params_info['verbose_period'])
            n_initial_explorations_run += 1
            # Increase the number of steps taken during the initial exploration
            n_events_et += learning_info['nsteps']

            # Update the sample size behind the calculation of the average reward
            # Note that we distinguish between the VERY FIRST iteration of this INI exploration (n_initial_exploration_run = 1)
            # and further iterations (which are run under the HARD killing case only when NO EXIT states from A are observed during the first INI exploration)
            # so that we do NOT use the currently stored sample size behind the average reward as a contribution to the new sample size when no estimation of the absorption set A
            # has been performed because, in that case, that sample size is associated to an average reward that corresponds to the average reward estimated
            # at the PREVIOUS policy learning step, which might correspond to a very different policy than the current one...
            # and we don't want to keep accumulating that sample size to the current policy learning step, also because this would give us an ALWAYS INCREASING sample size,
            # as the policy learning step increases, which does not make sense because that would imply that we are putting together on the same bag reward observations that
            # may come from totally different policies... from the very initial policy to the policy learned after many policy learning steps!
            if n_initial_explorations_run == 1:
                self.agent.getLearner().setSampleSizeForAverageReward( n_events_a + learning_info['nsteps'] )
            else:
                self.agent.getLearner().setSampleSizeForAverageReward( self.agent.getLearner().getSampleSizeForAverageReward() + learning_info['nsteps'] )

            # Check if EXIT states from A have been observed
            if len(learning_info['probas_stationary_exit_cycle_set']) == 0 and dict_params_simul['T'] < MAX_NUMBER_OF_STEPS_FOR_EXPECTATION:
                # Increase the simulation time for the initial exploration to see if by repeating the exploration, EXIT states can be observed
                dict_params_simul = update_number_of_steps_for_expectation(dict_params_simul, reason="before REPEATING the initial exploration (due to NO EXIT states observed from A)")

            flag_soft_killing_or_exit_states_observed_or_maximum_t_reached = dict_params_simul['soft_killing'] or \
                                                                             len(learning_info['probas_stationary_exit_cycle_set']) > 0 or \
                                                                             dict_params_simul['T'] == MAX_NUMBER_OF_STEPS_FOR_EXPECTATION

        n_absorption_cycles_used = learning_info['num_cycles']
        time_last_absorption = learning_info['last_cycle_entrance_time']

        print(f"Number of initial explorations run = {n_initial_explorations_run} for a total of {n_events_et} exploration steps. Last value of T = {dict_params_simul['T']}.")
        if not dict_params_simul['soft_killing']:
            # Manage the set of EXIT states from A, which define the selection of start states for the FV particles
            # The problem is that no EXIT states may have been observed in the above simulation, hence we need to recurse to a backup strategy.
            if len(learning_info['probas_stationary_exit_cycle_set']) > 0:
                print("--> SUCCESS: EXIT states from absorption set A observed! The EXIT states are stored as Activation Set in the learner for backup use in the future.")
                # Store the observed EXIT states as activation set in the learner, to be used as backup start states for FV when no EXIT states are observed for the SAME absorption set A
                self.agent.getLearner().setActivationSet(learning_info['probas_stationary_exit_cycle_set'], store_probabilities=True)
            else:
                # When NO EXIT states are observed, use the activation set stored in the learner as backup set of EXIT states keeping just the states that are OUTSIDE A (if any)
                # Note that, since the absorption set A CANNOT be reduced, any state in the backup exit set that is NOT in A, must be part of the external boundary of A.
                learning_info['probas_stationary_exit_cycle_set'] = self.agent.getLearner().getActivationSet(retrieve_probabilities=True)
                if len(learning_info['probas_stationary_exit_cycle_set']) > 0:
                    print("--> NO EXIT states from absorption set A were observed, but the set of EXIT states was restored to the Activation Set stored in the learner as backup,"
                          " using an uniform distribution (to achieve greater exploration):"
                          f"\n{learning_info['probas_stationary_exit_cycle_set']}")
                    # Find the backup exit states that are outside the current absorption set A and thus that are eligible to be selected as start states for the FV simulation
                    _backup_exit_states_outside_A = set(learning_info['probas_stationary_exit_cycle_set'].keys()).difference(dict_params_simul['absorption_set'])
                    print(f"of which the following states are OUTSIDE A:\n{_backup_exit_states_outside_A}")

                    # Create the EXIT state distribution from the backup set of EXIT states as a uniform probability on the states that are OUTSIDE A
                    # This allows two things:
                    # - Avoid failure of the assertion that checks the set of start states for the FV simulation contains states in the absorption set (done in _run_simulation_fv())
                    # - Distributing the start states homogeneously among the backup EXIT states outside A which could help balance a very skewed distribution from a previous excursion
                    # that may bias the concentration of particles in regions where they could get stuck.
                    learning_info['probas_stationary_exit_cycle_set'] = dict()
                    _proba_uniform = 1 / max(1, len(_backup_exit_states_outside_A))  # max(1, ...) in case the set of backup exit states that are outside A is empty
                    for s in _backup_exit_states_outside_A:
                        learning_info['probas_stationary_exit_cycle_set'][s] = _proba_uniform

                    # Store the backup set of EXIT states as activation set in the learner (so that it can be read at the next policy learning step)
                    self.agent.getLearner().setActivationSet(learning_info['probas_stationary_exit_cycle_set'], store_probabilities=True)

                    print(f"BACKUP set of EXIT states and their probability of selection as FV start states:"
                          f"\n{learning_info['probas_stationary_exit_cycle_set']}")
                    assert len(set(learning_info['probas_stationary_exit_cycle_set'].keys()).intersection(dict_params_simul['absorption_set'])) == 0, \
                        f"No states in the backup set of EXIT states must belong to the absorption set A, but the interesction of the backup set" \
                        f"\n{learning_info['probas_stationary_exit_cycle_set']}" \
                        f"\n and the absorption set" \
                        f"\n{dict_params_simul['absorption_set']}" \
                        f"\ngives {set(learning_info['probas_stationary_exit_cycle_set']).intersection(dict_params_simul['absorption_set'])}"

        # Store information about the estimated expected absorption time
        # One of the reasons for storing this information is to be able to use an estimated value for E(T_A) from a *previous* policy learning step,
        # in case an estimate happens to NOT be available at a particular learning step (due to not enough number of cycles observed).
        # So, the value we store here could be used in that case for the estimation of the long-run expected reward by FV.
        # Note that, if not enough cycles are observed during this initial exploration, we set the expected absorption time to the one measured in the previous policy learning step
        # whenever there is a previous step; if not, we set it to the number of steps taken during the initial exploration, which is our best estimate.
        if is_estimation_of_denominator_unreliable(): #n_absorption_cycles_used == 0:
            warning_msg = f"WARNING: The estimation of the expected absorption time E(T_A) cannot be reliably performed " \
                          f"because the number of cycles observed is zero or smaller than the minimum allowed: " \
                          f"{n_absorption_cycles_used} < {dict_params_simul['min_num_cycles_for_expectations']}"
            if True: #dict_params_info['t_learn'] == 0:
                # DM-2025/07/14: We now ALWAYS set E(T_A) to the number of simulation steps when no cycles are observed because the previous strategy of setting it to the
                # value estimated at the previous policy learning step could have been BADLY UNDERESTIMATED (because based on very different settings, different A and policy)
                # giving a very large estimated expected reward (because denominator of FV ratio too small), and this was responsible for making the policy go SNAFU under
                # the execution parameters of STORED=True, FIXED=True.
                expected_absorption_time = float(np.nan_to_num(learning_info['expected_cycle_time']))
                _censored_time = learning_info['nsteps'] - time_last_absorption
                assert _censored_time >= 0
                expected_absorption_time += (_censored_time - expected_absorption_time) / (n_absorption_cycles_used + 1)
                warning_msg += f"\nThe estimation of E(T_A) will INCLUDE the number of steps taken between the last entry time to A and the number of simulation steps run: {expected_absorption_time}"
                _case_expected_absorption_time = "estimated using the last censored observation"
            else:
                expected_absorption_time = self.agent.getLearner().getExpectedAbsorptionTime()
                warning_msg += f"\nThe value of E(T_A) will be set to the value of the previous policy learning step: {expected_absorption_time}"
                _case_expected_absorption_time = "set to the value estimated at the previous learning step"
            print(warning_msg)
            warnings.warn(warning_msg)
        else:
            expected_absorption_time = learning_info['expected_cycle_time']
            _case_expected_absorption_time = "estimated at this learning step on fully observed cycles"
        assert not np.isnan(expected_absorption_time) and expected_absorption_time is not None, \
            f"The expected cycle time estimated by the initial exploration must not be NaN nor None: {expected_absorption_time}"
        self.agent.getLearner().setExpectedAbsorptionTimeAndNumCycles(expected_absorption_time, n_absorption_cycles_used)
        print(f"--> Estimated absorption time E(T_A) on {n_absorption_cycles_used} cycles: {expected_absorption_time} ({_case_expected_absorption_time})")

        # Store information about the average reward observed during the initial exploration of the environment
        # One of the reasons for storing this is that we can analyze how much the initial exploration and the FV simulation contribute to the final average reward.
        # Note that it is NOT the same to compute the average reward as a simple average of the rewards observed during the excursion and to retrieve it directly from the
        # average reward stored in the learner, because the latter may be the result of a correction of an initial estimate of the average reward
        # (e.g. taken from a previous policy learning step when use_average_reward_stored_in_learner=True) whereas the former is the direct average on the rewards
        # observed EXCLUSIVELY on this particular excursion.
        average_reward_from_initial_exploration = np.mean(self.agent.getLearner().getRewards())
        self.agent.getLearner().setAverageRewardInitialExploration(average_reward_from_initial_exploration)
        print(f"--> [t_learn={dict_params_info['t_learn']}] Average reward estimated from the initial exploration: {average_reward_from_initial_exploration} (it will be used to correct the value functions estimated by the FV simulation)")

        # When reward shaping is used (i.e. the entry 'reward_for_exit_states' is not None) to promote the exploration of the states at the boundary of A,
        # we should update the policy AFTER the initial exploration, because the value functions are reset at the beginning of the FV simulation
        # (they are reset so that the FV simulation reflects the original environment, WITHOUT reward shaping).
        # This is important, because o.w. the policy for all the states in A will never change, even if we do reward shaping.
        if dict_params_simul['reward_for_exit_states'] is not None:
            print(f"[AFTER INITIAL EXPLORATION] Updating the policy based on the value functions learned with reward shaping on EXIT states from A (r={dict_params_simul['reward_for_exit_states']})")
            dict_params_simul['learner_policy'].learn_natural(self.agent.getLearner().getA().getValues())

        #-- Step 2: Simulate N particles with Fleming-Viot to compute the empirical distribution and estimate the stationary probabilities, and from them the expected reward
        print("\n*** FLEMING-VIOT SIMULATION ***")
        if is_estimation_of_denominator_unreliable():
            warning_msg = f"WARNING: [t_learn={dict_params_info['t_learn']}] The Fleming-Viot estimate of the long-run expected reward may be unreliable because the estimation of the expected absorption time E(T_A) " \
                          f"is based on too few reabsorption cycles: {n_absorption_cycles_used} < {dict_params_simul['min_num_cycles_for_expectations']}"
            print(warning_msg)
            warnings.warn(warning_msg)
        # Do not run the FV simulation when NO exit states from A have been observed and no backup set of exit states is available
        # Reason: FV needs to know where to distribute the FV particles and this location should be in the outside boundary of A,
        # which is estimated by the EXIT states from A observed during the E(T_A) simulation.
        if not dict_params_simul['soft_killing'] and len(learning_info['probas_stationary_exit_cycle_set']) == 0:
            # This is the case when the start states of the FV particles cannot be chosen because:
            # - we are NOT using the soft killing strategy
            # - no EXIT states from A were observed during the initial exploration of the environment and no backup set is available
            state_counts_all = state_counts_et
            expected_reward = self.agent.getLearner().getAverageReward()
            probas_stationary = dict()
            expected_absorption_time = np.nan
            max_survival_time = np.nan
            n_events_a = n_events_a
            n_events_et = n_events_et
            n_events_fv = 0

            warning_msg = f"*** WARNING: *** The set of EXIT states from A is EMPTY!!! The FV simulation will not be run. " \
                          f"The estimate of the long-run expected reward will be set to the estimate obtained during the initial exploration: {expected_reward}"
            print(warning_msg)
            warnings.warn(warning_msg)

            #input("Press ENTER to continue...")

            assert dict_params_simul['T'] == MAX_NUMBER_OF_STEPS_FOR_EXPECTATION and self.agent.getLearner().getNumTimeStepsForExpectation() == MAX_NUMBER_OF_STEPS_FOR_EXPECTATION, \
                f"The number of steps for the initial exploration must have reached its possible maximum of T={MAX_NUMBER_OF_STEPS_FOR_EXPECTATION} when no EXIT states were observed" \
                f"during the several trials of initial exploration of the environment: T={self.agent.getLearner().getNumTimeStepsForExpectation()}"
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
            # However, we would need the average reward if we estimate the advantage function in a better way,
            # namely as R(n+1) - avg.R(n) + V(S(n+1)) - V(S(n)) at each simulation step n.
            expected_absorption_time = self.agent.getLearner().getExpectedAbsorptionTime()
            print(f"FV simulation on N={N} particles starts...")
            # TEMPORARY (2023/12/05) To test whether FV learns the optimal policy when the values of the states and actions taken in set A are NOT estimated at all
            #self.agent.getLearner().reset(reset_value_functions=True)
            #print(f"VALUE FUNCTIONS RESET BEFORE STARTING FLEMING-VIOT!\nV = \n{self.agent.getLearner().getV().getValues()}\nQ = \n{self.agent.getLearner().getQ().getValues()}")
            # TEMPORARY
            if self.agent.getLearner().getLearningTask() == LearningTask.CONTINUING:
                # Continuing learning task. Note that the learning criterion can be either AVERAGE or DISCOUNTED, but the former is more common.
                method_fv = self._run_simulation_fv; uniform_jump_rate = N  # We need to divide the FV sum by N in order to make it comparable with the denominator E(T_A) whose time measure is N times shorter than the time measure of an FV particle (as the FV particle is chosen on average once every N time steps of the FV system)
                #method_fv = self._run_simulation_fv_fraiman; uniform_jump_rate = 1  # In this case, all FV particles are updated at the same system's time step, therefore no adjustmend needs to be done to the FV sum.
                #method_fv = self._run_simulation_fv_fraiman_modified; uniform_jump_rate = 1  # In this case, all FV particles are updated at the same system's time step, therefore no adjustmend needs to be done to the FV sum.
                start_set = None  # Normally we should use `probas_stationary_start_state_fv` to select the start states in FV
                if probas_stationary_start_state_fv is None:
                    # Define the stationary probability for the start state in the FV simulation to be carried out below.
                    # This distribution is derived from the initial exploration run above, and it depends on whether SOFT KILLING is used or not, as follows:
                    # - if SOFT KILLING: it is set to the stationary distribution of the state at which a killing CLOCK occurs.
                    # - Otherwise: it is set to the stationary exit probability from the absorption set.
                    # *** IMPORTANT: Note that this distribution is NOT stored in the FV learner ***
                    # In fact, if we stored it in the FV learner and this simulation is part of a policy learning process, the next time the simulation is called
                    # (e.g. at a subsequent policy learning step) the start distribution for the FV particles would no longer be None,
                    # because the distribution (probas_stationary_start_state_fv, which is checked against `None` above)
                    # would be read from the distribution stored in the FV learner by the caller, and this would impede an update of the start distribution
                    # (which is required when the policy had been updated --and possibly the absorption set A as well-- at the latest policy learning step).
                    if dict_params_simul['soft_killing']:
                        # DM-2025/01/05: Starting at the start of a cycle is not so effective because the start of a cycle is a frequently visited state and thus the FV particles
                        # remain mostly in the set of frequently visited states. So, it is better to start at the state JUST BEFORE the killing state
                        # (collected at each occurrence of a killing event).
                        #probas_stationary_start_state_fv = learning_info['probas_stationary_start_cycle']
                        probas_stationary_start_state_fv = learning_info['probas_stationary_end_cycle']
                        assert len(probas_stationary_start_state_fv) > 0, "(Assertion added 22-May-2025) The number of end cycle states in the SOFT killing case must not be empty"
                        start_state_selection_case = "1 - END CYCLE STATES"
                    else:
                        probas_stationary_start_state_fv = learning_info['probas_stationary_exit_cycle_set']
                        assert len(probas_stationary_start_state_fv) > 0, "(Assertion added 22-May-2025) The probability distribution of EXIT states must not be empty"
                        start_state_selection_case = "2 - EXIT CYCLE STATES"
                    start_state_selection_case += f" (size: {len(probas_stationary_start_state_fv)})"
                    print(f"[Analysis of EXIT states] => START STATE selection case: {start_state_selection_case}")

                    # Either when soft killing is used or not, set the distribution for the start state for the E(T_A) simulation
                    # also to the stationary distribution for the FV start state. This MUST be the case in the soft killing case.
                    # However, in the HARD killing, this is also useful because such distribution will be used to select the start state of the initial exploration
                    # that is used to estimate E(T_A) --which has been ALREADY RUN in the CURRENT policy learning step-- there will be higher chances that the entrance to A
                    # follows the stationary entry distribution than selecting the start state uniformly at random.
                    # (note that we cannot use this distribution to select the start state for the estimation of E(T_A) in the CURRENT policy learning step
                    # because that simulation has ALREADY BEEN RUN, which is in fact the simulation that has been used to estimate the stationary EXIT distribution from A,
                    # just used (normally) to define the distribution for the start state of the FV particles --read above from learning_info['probas_stationary_exit_cycle_set']).
                    self.getAgent().getLearner().setProbasStationaryStartStateET(probas_stationary_start_state_fv)
            else:
                # EPISODIC learning task. (2025/05/26) FV is not really prepared for this situation at this point UNLESS we use it simply as an oversampling mechanism.
                # To leverage the Fleming-Viot oversampling mechanism in learning a policy (and make learning independent of the estimated expected reward by Fleming-Viot),
                # follow the directives described in the error message triggered before when the user chooses the EPISODIC learning task.
                raise ValueError("The EPISODIC learning tasks is currently NOT handled by Fleming-Viot. "
                                 "If you wish to use Fleming-Viot in episodic tasks, just call it with the usual way, i.e. CONTINUING task with either AVERAGE or DISCOUNTED reward criterion "
                                 "and REQUEST that the policy learner do NOT use the advantage function estimated by the critic learning process to update the policy "
                                 "(which uses the average reward estimated by Fleming-Viot to correct each observed reward and thus construct the differential value function) "
                                 "and to instead use the advantage function computed as `Q - V`, where the average reward correction term persent in both Q and V cancels out."
                                 "\nAt the moment of writing (25-May-2025), this behaviour can be accomplished in test/tests.py by setting `use_advantage = False` "
                                 "a variable that is passed to the online policy learner or otherwise parsed before applying NPG learning (Natural Policy Gradient), "
                                 "in case the latter is the learning method chosen for the policy learner.")
                # When running FV to learn the value functions V(s) and Q(s,a) under the EPISODIC learning task
                # (which implies a DISCOUNTED learning criterion, as the AVERAGE reward criterion cannot be used in that case because of the random episode lengths),
                # the particles should start all over the place outside A, so that we can explore all those states
                # more uniformly, as we need to estimate the value of EACH state.
                # I've tested starting all the particles at the outer boundary of A to estimate V(s) on a 5-state 1D gridworld with +1 reward with a very small
                # stationary probability due to a policy that goes left with 90% probability, and we get:
                # - More variability in the estimation of the state value V(s) across replications.
                # - A few replications (less than 10%) estimate V(s) = 0 for some states s, which does NOT happen when starting the particles uniformly distributed outside A.
                # Parameters: N=50 particles, R=20 replications.
                # See meeting minutes in entry dated 17-Jan-2024 for more details.
                method_fv = self._deprecated_run_simulation_fv_discounted; uniform_jump_rate = 1
                start_set = self.agent.getLearner().active_set.difference(self.env.getTerminalStates())
            # DM-2025/07/13: When use_average_reward_stored_in_learner=True we should do precisely that, use the average reward stored in the learner
            # and this is what is retrieved by self.agent.getLearner().getAverageReward(), NOT by estimated_average_reward_before_initial_exploration, as was the case before.
            # This is why this line was replaced with the one that follows.
            #estimated_average_reward_before_fv = estimated_average_reward_before_initial_exploration  if use_average_reward_stored_in_learner and dict_params_info['t_learn'] > 0 and estimated_average_reward_before_initial_exploration is not None and not np.isnan(estimated_average_reward_before_initial_exploration) \
            #                                                                                          else average_reward_from_initial_exploration if dict_params_simul['reward_for_exit_states'] is None \
            #                                                                                          else None
            estimated_average_reward_before_fv = self.agent.getLearner().getAverageReward() if use_average_reward_stored_in_learner and dict_params_simul['reward_for_exit_states'] is None \
                                                                                            else average_reward_from_initial_exploration if dict_params_simul['reward_for_exit_states'] is None \
                                                                                            else None
            n_events_fv, state_values, action_values, advantage_values, state_counts_fv, phi, df_proba_surv, expected_absorption_time, max_survival_time, absorption_set, less_frequently_visited_set = \
                method_fv(  dict_params_info['t_learn'], envs,
                            dict_params_simul['absorption_set'],
                            start_set=start_set if not dict_params_simul['soft_killing'] else None,
                            max_time_steps=dict_params_simul['max_time_steps'],
                            max_time_steps_for_absorbed_particles_check=dict_params_simul['max_time_steps_for_absorbed_particles_check'],
                            min_prop_absorbed_particles=dict_params_simul['min_prop_absorbed_particles'],
                            stopping_criterion_fv=dict_params_simul['stopping_criterion_fv'],
                            dist_proba_for_start_state=probas_stationary_start_state_fv,
                            start_state_selection_case=start_state_selection_case,
                            soft_killing=dict_params_simul['soft_killing'],
                            dict_proba_killing=dict_params_simul['proba_killing'] if dict_params_simul['soft_killing'] else None,
                            update_absorption_set_with_fv_visits=dict_params_simul['update_absorption_set_with_fv_visits'],
                            reward_shaping=dict_params_simul['reward_for_exit_states'] is not None and absorption_set_has_been_updated,
                            expected_absorption_time=expected_absorption_time,
                            # IMPORTANT: (2024/08/11) Using a previously estimated average reward as initial estimate of the average reward estimation by FV
                            # ASSUMES that that initial estimate only contains reward information from OUTSIDE the absorption set A!
                            # In fact, if this were not the case, we would be using that average reward value to update the average reward estimated by FV involving ONLY
                            # states that are OUTSIDE A. Therefore we would be "contaminating" rewards OUTSIDE A with reward information coming from A... and that would be wrong.
                            # TODO: (2024/08/11) Pass to the FV estimator of the average reward an initial estimation of the average reward that includes ONLY contributions from rewards observed OUTSIDE A (as explained above) (as only THAT piece of information of the average reward should be used to update the average reward that is estimated by FV (namely the reward outside A)
                            # 2025/01/19: Use the average reward estimated from the initial exploration as the estimated_average_reward value to test the REWARD SHAPING OF THE EXIT STATES (because in the current implementation, most of the reward defined by reward shaping is NOT seen by the FV simulation and thus the value of `estimated_average_reward_before_initial_exploration` will not have their full contribution)
                            #estimated_average_reward=average_reward_from_initial_exploration,
                            #estimated_average_reward=0.0,  # (2025/05/18) Use this to the ABLATION study of estimating the average reward at a wrong value (e.g. always 0)
                            # LOGIC FOR THE estimated_average_reward PARAMETER:
                            # 1) IF this is the first policy learning step:
                            #   - IF reward shaping has been used during the initial exploration (to boost the visit of EXIT states)
                            #     (i.e. `dict_params_simul['reward_for_exit_states'] is not None`)
                            #       => set the estimated_average_reward=None (because we don't want to bias the FV learning based on information obtained from a changed environment)
                            #   - ELSE:
                            #       => use the average reward estimated during the initial exploration of the environment (average_reward_from_initial_exploration)
                            # 2) ELSE (i.e. this is NOT the first policy learning step):
                            #   - IF the user requested to use the average reward estimated at the previous learning step:
                            #       => use such average reward value if if not missing (estimated_average_reward_before_initial_exploration, which actually should actually only take the value None and never np.nan... but still we check that here to be sure, as adding an assertion is not possible)
                            #   - ELSE:
                            #       => use the average reward estimated during the initial exploration of the environment (average_reward_from_initial_exploration)
                            estimated_average_reward=estimated_average_reward_before_fv,
                            use_fixed_average_reward=use_fixed_average_reward,
                            keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=keep_fv_estimation_of_average_reward_and_stationary_probability_consistent,
                            epsilon_random_action=dict_params_simul['epsilon_random_action'],
                            seed=dict_params_simul['seed'] + 131713,    # Choose a different seed from the one used by the single Markov chain simulation (note that this seed is the base seed used for the seeds assigned to the different FV particles)
                            verbose=dict_params_info['verbose'],
                            verbose_period=dict_params_info['verbose_period'],
                            plot=dict_params_info['plot'],
                            colormap=dict_params_info['colormap'],
                            pause=dict_params_info['pause'])

            #-- Extend the absorption set stored in the FV learner, based on the state visit frequency during the FV simulation
            _states_in_absorption_set_with_nonzero_reward = [s for s in absorption_set if
                                                             self.env.getReward(self.env.getStateFromIndex(s, simulation=True)) != 0.0]
            assert len(_states_in_absorption_set_with_nonzero_reward) == 0, f"[CHECK #2] The absorption set must not contain states with non-zero reward. The following states in the absorption set have non-zero reward: {_states_in_absorption_set_with_nonzero_reward}"

            if  dict_params_simul['soft_killing'] or \
                (dict_params_simul['update_absorption_set_with_fv_visits'] and dict_params_simul['T'] < MAX_NUMBER_OF_STEPS_FOR_EXPECTATION):
                # Increase the absorption set A based on visit frequencies during the FV simulation,
                # when soft killing is used, and otherwise ONLY when the number of steps T used for the initial simulation has NOT reached its maximum value.
                # Reason: the fact that T reached the maximum allowed value is an indicator that the absorption set has grown too much...
                # ALSO: this condition also makes the increase decision of the absorption set INDEPENDENT OF THE AGENT KNOWING THE SIZE OF THE ENVIRONMENT,
                # which is supposed to NOT know (i.e. the agent should not know anything about the environment apart from what it finds during exploration).
                _absorption_set_has_been_updated, _number_of_new_states_in_absorption_set = update_absorption_set_if_not_too_large(absorption_set)
                if _absorption_set_has_been_updated:
                    # Update also the set of less frequently visited states during the FV simulation in order to keep consistency of this piece of information
                    # with the information about the absorption set.
                    self.agent.getLearner().setLessFrequentlyVisitedSet(less_frequently_visited_set)
            else:
                print(f"Absorption set NOT updated because the number of steps T for the expected reabsorption time has reached its maximum of {MAX_NUMBER_OF_STEPS_FOR_EXPECTATION}.")
                    
            #-- Process the estimated expected reward
            state_counts_all = state_counts_et + state_counts_fv
            #print(f"Shape of proba surv and phi: {df_proba_surv.shape}")
            print("Expected reabsorption time E(T_A): {:.3f} ({} cycles)".format(expected_absorption_time, learning_info['num_cycles']))
            print(f"proba_surv P(T>t):\n{df_proba_surv}")
            print(f"Time-average Phi value per state of interest:")
            if len(phi) <= 50:
                _average_phi_values = dict([(x, np.mean(phi[x]['Phi'])) for x in phi.keys()])
                print(_average_phi_values)
            else:
                print(f"(not printed because too many states of interest x in Phi(t, x) ({len(phi)})")
            print(f"Max survival time: {max_survival_time}")

            assert expected_absorption_time is not None and expected_absorption_time > 0, f"The expected reabsorption time must be given and be positive ({expected_absorption_time})"
            if dict_params_simul['stopping_criterion_fv'] in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES]:
                # When not using any condition on the absorbed particles for stopping the FV simulation,
                # the expected reward and stationary probabilities have NOT yet been computed (e.g. iteratively by the FV simulation)
                # => Compute these quantities now using a PARAMETRIC estimator of P(T>t) so that ALL the observation range of Phi(x, t) is used
                probas_stationary, integrals = estimate_stationary_probabilities(phi, df_proba_surv,
                                                                                 expected_absorption_time,
                                                                                 parametric=True,
                                                                                 uniform_jump_rate=uniform_jump_rate)
                # Expected reward on states outside A only, i.e. this is the expected reward estimated by FV
                # If we want to compute the expected reward on ALL states of the environment, we should add the estimated expected reward estimated in A
                # which is part of the expected reward estimated during the initial exploration but which is currently NOT split into "inside A" and "outside A"
                # as is needed if we want to add the estimated expected reward on states in A to the estimated expected reward on states OUTSIDE A.
                # TODO: (2025/05/30) Implement the appropriate estimation of the expected reward that takes into account rewards observed "inside A" and "outside A" (as just described). See also the ELSE block below for more comments and ideas.
                expected_reward_fv = compute_expected_reward(envs[0], probas_stationary)

                if keep_fv_estimation_of_average_reward_and_stationary_probability_consistent:
                    expected_reward = expected_reward_fv
                else:
                    # We update the average reward estimated before running FV with the FV reward considered as innovation
                    expected_reward = estimated_average_reward_before_fv + (expected_reward_fv - estimated_average_reward_before_fv) / 2
                        ## The `/2` division comes from the assumption done in the update_average_reward() function in _run_simulation_fv()
                        ## which makes the iterative update of the average reward be computed as:
                        ##      avgR(ini) + (avgR(FV) - avgR(ini)) * n_survival_times_observed_so_far / (N + n_survival_times_observed_so_far))
                        ## and observing that at this stage, where the FV simulation has ended and all N particles have been considered in the computation of FV average reward
                        ## (either because they have ALL ben absorbed at least once or because the non-absorbed particles have been forcedly absorbed as "censoring" absorptions)
                        ## we happen to have that the two quantities involved in the factor appearing at the end coincide, i.e. n_survival_times_observed_so_far = N.

                # Store the expected reward as average reward in the learner object
                # so that we can retrieve the average reward estimated by FV by using the method GenericLearner.getAverageReward()
                self.agent.getLearner().setAverageReward(expected_reward)
            else:
                # When looking at the number of absorbed particles for stopping the FV simulation,
                # The expected reward and stationary probabilities have been computed ITERATIVELY by the learner during the FV simulation
                # => Retrieve this information from the learner
                assert dict_params_simul['stopping_criterion_fv'] in [StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES, StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN]
                integrals = self.agent.getLearner().getIntegral()
                probas_stationary = dict.fromkeys(integrals, 0.0)
                for _state, _integral_for_state in integrals.items():
                    probas_stationary[_state] = _integral_for_state / uniform_jump_rate / expected_absorption_time
                # Note that the expected reward retrieved below may or may not contain information about the rewards in A.
                # This depends on what the _run_simulation_fv.update_average_reward() function does at the end of the simulation,
                # i.e. when calculating the FINAL average reward computed by the FV excursion:
                # - If the function only considers the rewards observed outside A to compute the FINAL average reward
                # (keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=True), then no reward from A is included in the expected reward retrieved below.
                # - If the function blends the average reward estimated from the initial exploration into the FINAL average reward
                # (keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=False), then the rewards observed in A are included.
                #
                # PERHAPS a better approach for an estimation of the expected reward that takes into account both the rewards received in A and those outside A is
                # to split the rewards observed during the initial exploration into "inside A" and "outside A" and then use "outside A" part to blend it into the FV
                # estimation of the expected reward outside A, and finally SUM the two rewards, "inside A" and "outside A",
                # possibly a sum weighted by the sample sizes behind their calculation (although the sample size behind the calculation of the average reward outside A
                # may not be clear due to the FV estimation, which doesn't have a clear sample size involved (or perhaps, the number of particles could be used as sample size?))
                expected_reward = self.agent.getLearner().getAverageReward()

            ##### TEMPORARY PATCH (WORKED!) (TO TRY TO FIX THE UNLEARNING OF THE POLICY IN THE MOUNTAIN CAR PROBLEM) #########
            # If the expected reward estimated by FV is 0.0, DISCARD IT and SET IT TO the expected reward estimated by the initial exploration
            # (UNLESS reward shaping was performed during the initial exploration of the environment to promote a policy that takes the agent towards the boundary of A,
            # in which case we should discard any average reward estimated from that phase because it corresponds to a modified environment).
            if expected_reward == 0.0 and dict_params_simul['reward_for_exit_states'] is None:
                expected_reward_to_restore = average_reward_from_initial_exploration
                self.agent.getLearner().setAverageReward(expected_reward_to_restore)
                expected_reward = expected_reward_to_restore
                print(f"***** ESTIMATED EXPECTED REWARD BY THE FV SIMULATION PROCESS IS ZERO!"
                      f"\n***** => The expected reward estimated during the initial exploration of the environment has been restored in the learner! (restored reward = {expected_reward})")
            ##### TEMPORARY PATCH (WORKED!) (TO TRY TO FIX THE UNLEARNING OF THE POLICY IN THE MOUNTAIN CAR PROBLEM) #########

            if True or DEBUG_ESTIMATORS or show_messages(dict_params_info['verbose'], dict_params_info['verbose_period'], dict_params_info['t_learn']):
                #max_rows = pd.get_option('display.max_rows')
                #pd.set_option('display.max_rows', None)
                #print("Phi(t):\n{}".format(phi))
                #pd.set_option('display.max_rows', max_rows)
                print(f"Integrals: (NOT adjusted by the uniform jump rate of {uniform_jump_rate}, which should divide this value)")
                if len(integrals) <= 20:
                    print(integrals)
                else:
                    print(f"(not printed because too many states of interest x in Integral(x): {len(integrals)})")
                print("[t_learn={}] Expected reabsorption time (on {} cycles): {}".format(dict_params_info['t_learn']+1, learning_info['num_cycles'], expected_absorption_time))
                print("Stationary probabilities: {}".format(probas_stationary))
                print("Expected reward = {}".format(expected_reward))
                print("Average reward stored in learner = {}".format(self.agent.getLearner().getAverageReward()))
                ##### TEMPORARY PATCH (comment assertion when applying patch) #####
                #assert np.isclose(expected_reward, self.agent.getLearner().getAverageReward()), \
                #    f"The average reward estimated by FV ({expected_reward}) must be equal to the average reward stored in the FV learner ({self.agent.getLearner().getAverageReward()})"
                ##### TEMPORARY PATCH (comment assertion when applying patch) #####

        return state_values, action_values, advantage_values, state_counts_all, state_counts_et, probas_stationary, expected_reward, expected_absorption_time, n_absorption_cycles_used, \
               time_last_absorption, max_survival_time, n_events_a, n_events_et, n_events_fv

    def run_exploration(self, t_learn=-1, max_time_steps=1000, epsilon_random_action=0.0, seed=None, verbose=False, verbose_period=1):
        """
        Performs an exploration of the environment under a CONTINUING learning task, without value functions learning,
        just with the purpose of collecting state visit frequencies and returning the observed average reward.
        The value of the average reward is NOT updated in the agent's learner.

        This is typically used to estimate the absorption set A to use in FV learning or the killing probability in the case of soft killing,
        and to estimate the expected reward to be used as correction value for a future exploration that is used for value functions learning
        under the average reward criterion.

        The learner stored in self.agent is used to store the trajectory (but NOT the average reward), which is assumed to have the following methods defined:
        - reset()
        - update_trajectory()
        This is the case if the learner is an instance of the GenericLearner class.

        t_learn: (opt) int
            The learning step number (starting at 0) for which the simulation is run when learning is used in the context of policy learning.
            This is ONLY used for informational purposes, e.g. to show which stage of the policy learning we are at.
            default: -1 (which means that the simulation is not part of a policy learning process by default)

        max_time_steps: (opt) int
            Number of steps to run.
            default: 1000

        epsilon_random_action: (opt) float in [0, 1]
            Probability of taking a random action at each step.
            default: 0.0

        seed: (opt) int
            Seed to use for the random number generator for the simulation, both for choosing the agent's next action and the next state of the environment
            given the chosen action.
            default: None

        verbose: (opt) bool
            Whether to show a message with the transition of the system at a frequency defined by the `verbose_period` parameter.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Tuple with the following three elements:
        - learner object inheriting from class GenericLearner: the learner object stored in self.agent where the observed trajectory is stored.
        - the number of steps used by the simulation.
        - the average reward observed during the simulation
        """
        if max_time_steps is None or max_time_steps < 0 or max_time_steps == +np.Inf:
            raise ValueError(f"Parameter `max_time_steps` must be a positive finite number: {max_time_steps}")

        policy = self.getAgent().getPolicy()
        learner = self.getAgent().getLearner()

        # Set seeds of:
        # - torch --> responsible for selecting the action when the policy is modeled via a neural network.
        # - the policy's environment --> just in case this environment does NOT have the same memory address as self.env
        #   (e.g. when policies are deepcopied from the original policy in order to compare different learning methods)
        # - numpy --> responsible for deciding whether a (completely) random action is chosen (when epsilon_random_action > 0).
        #   numpy is used to both draw a random number to decide whether to choose a random action (without following the policy)
        #   and then to actually choose that random action, if this ends up being the case.
        # - the object's environment --> responsible of deciding on the next state given the action.
        if seed is not None:
            torch.manual_seed(seed)
            policy.env.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        # Reset the environment to a state according to its initial state distribution
        self.env.reset()

        # Reset the learner in order to reset any trajectory potentially stored in the learner
        # Note that we reset the learner and NOT simply the trajectory stored in it because the reset() method defines a few attributes
        # that are not defined by the `learner` constructor which should also be set before starting a simulation.
        # (THIS SHOULD BE CHANGED PERHAPS --by defining all the attributes that are necessary during a simulation in the CONSTRUCTOR of the learner NOT only by its reset() method
        # --but I am not really sure about this need, because a learner should be anyway RESET before starting a simulation... right?)
        # Note that we do NOT reset the value functions nor the average reward because these are not learned nor updated in the learner object
        learner.reset(reset_episode=True)

        # Initialize the rewards list with a dummy first element as if the environment came from an action taken by the agent and a reward of LANDING at the reset state is observed
        # This is important for the correct association of states and rewards in the sense that learner.states[k] is the state where the reward learner.rewards[k] is observed
        # once the state stored in learner.states[k] is visited, and this is particularly useful if we estimate the absorption set A in the FV simulation
        # with function compute_set_of_frequent_states_with_zero_reward() where the lists of observed states and rewards are passed to the function which are expected
        # to be aligned as just indicated! (because e.g. we filter on the states receiving zero reward).
        # (Note that this alignment corresponds to the usual RL notation of a trajectory as S(0), A(0), R(1), S(1), A(1), ..., where R(1) corresponds to the reward received
        # when the system transitions from state S(0) TO S(1) after taking action A(0) or, otherwise said, when the system LANDS / transitions-TO state S(1), that is the index
        # of S(t) at which the reward R(1) is received is t = 1 (as opposed to t = 0, in which case R(1) would be the reward observed when transitioning from S(0), i.e. by the
        # fact that the system leaves S(0)... but this is not the convention used here).
        # IMPORTANT: This learner.rewards attribute is the attribute present in the GenericLearner class, NOT in the Learner class of which the `learner` here is an instance of.
        # The Learner class stores rewards in learner._rewards which DOES already have ONE element at the beginning with the reward associated to the start state of the episode.
        # Recall that the Learner class stores just the trajectory observed WITHIN the episode, whereas the GenericLearner class stores the WHOLE trajectory, e.g. the trajectory
        # observed along ALL the episodes run.
        # Here we are interested in tracking the trajectory observed over ALL episodes, therefore we will look at the attributes of GenericLearner and not at the Learner attributes
        # This is why we now explicitly add the first reward (received when visiting the initial state) to the learner.rewards attribute, which does NOT yet have it stored.
        learner.rewards += [self.env.getReward(self.env.getState())]

        t = 0               # Step counter: the first step is 1, as t represents the time at which the Markov chain transitions to the NEXT state. See more details at the @note at the beginning of the file.
        t_episode = -1      # Step counter within episode: the first step is 0, as t_episode indexes the step BEFORE transition so that we can write S(0), A(0), R(1), S(1), A(1), ...
        done_episode = self.env.getState() in self.env.getTerminalStates()
        total_reward = 0.0  # Variable that is used to compute the continuing (as opposed to episodic) average reward observed during the exploration.
                            # Note that we do NOT use a call to update_average_reward() following the GenericLearner.update_trajectory() method below
                            # (as the simulation proceeds) because that call would call the Learner.update_average_reward() method (because the learner object
                            # is typically an instance of that class), which is actually used to update the average reward stored in the learner ONCE AN EPISODE has ended,
                            # and thus it involves combining the episodic average reward with the continuing average reward stored in the learner in a particular way...
                            # And we don't want to enter into that hassle here, but instead simply compute the continuing average reward of the process, regardless of
                            # the episodic average reward.
                            # By doing this ad-hoc computation of the average reward, we keep its computation simple.
        while t < max_time_steps:
            t += 1
            t_episode += 1

            state = self.env.getState()

            if done_episode:
                # We have reached a terminal state
                # => Reset the environment
                action = 0
                next_state = self.env.reset()
                reward = self.env.getReward(next_state)
                done_episode = next_state in self.env.getTerminalStates()
                t_episode = -1
            else:
                action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
                next_state, reward, done_episode, info = self.env.step(action)

            # Update the trajectory stored in the learner
            learner.update_trajectory(t, state, action, reward)
            total_reward += reward

            if show_messages(verbose, verbose_period, t):
                print(f"t: {t}, t in episode: {t_episode}, s={state}, a={action} -> ns={next_state}, r={reward}")

        # Finalize the trajectory so that the `states`, `actions` and `rewards` list are of the same length
        learner.times += [t+1]
        learner.states += [next_state]
        learner.actions += [np.nan]
        assert len(learner.rewards) == t + 1     # This assertion is to corroborate that we need to divide total_reward by t+1, as that is tantamount to doing np.mean(learner.rewards)
        average_reward = total_reward / (t+1)    # We need to divide by the length of learner.times because if at the start state there is a reward it is counted (see Learner._reset_trajectory() method)
        assert len(learner.states) == len(learner.actions) and len(learner.actions) == len(learner.rewards)
        if len(learner.times) > 1:
            assert learner.times[-1] == learner.times[-2] + 1

        return learner, t, average_reward

    def run_exploration_and_learn_value_functions(self, t_learn=-1, max_time_steps=1000, epsilon_random_action=0.0, seed=None, verbose=False, verbose_period=1):
        """
        Performs an exploration of the environment under a CONTINUING learning task, with the main objective of collecting state visit frequencies.
        However, the exploration is also used to learn the differential value functions.

        The learner stored in self.agent is used to learn and store the trajectory, which is assumed to have the following methods defined:
        - reset()
        - learn()

        EXAMPLE OF CALLING THIS METHOD:
        learner, n_events_absorption_set_estimation = self.run_exploration_and_learn_value_functions(t_learn=0, max_time_steps=dict_params_simul['T'], seed=1313)

        t_learn: (opt) int
            The learning step number (starting at 0) for which the simulation is run when learning is used in the context of policy learning.
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.
            default: -1 (which means that the simulation is not part of a policy learning process by default)

        max_time_steps: (opt) int
            Number of steps to run.
            default: 1000

        seed: (opt) int
            Seed to use for the random number generator for the simulation, both for choosing the agent's next action and the next state of the environment
            given the chosen action.
            default: None

        verbose: (opt) bool
            Whether to show a message with the transition of the system at a frequency defined by the `verbose_period` parameter.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: Tuple
        Tuple with the following two elements:
        - learner: the learner object inheriting from class Learner where the trajectory has been stored.
        - t: the number of steps used by the simulation.
        """
        if max_time_steps is None or max_time_steps < 0:
            raise ValueError(f"Parameter `max_time_steps` must be a positive number: {max_time_steps}")

        policy = self.getAgent().getPolicy()
        learner = self.getAgent().getLearner()

        # Set seeds of:
        # - torch --> responsible for selecting the action when the policy is modeled via a neural network.
        # - the policy's environment --> just in case this environment does NOT have the same memory address as self.env
        #   (e.g. when policies are deepcopied from the original policy in order to compare different learning methods)
        # - numpy --> responsible for deciding whether a (completely) random action is chosen (when epsilon_random_action > 0).
        #   numpy is used to both draw a random number to decide whether to choose a random action (without following the policy)
        #   and then to actually choose that random action, if this ends up being the case.
        # - the object's environment --> responsible of deciding on the next state given the action.
        if seed is not None:
            torch.manual_seed(seed)
            policy.env.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        # Reset the environment to a state according to its initial state distribution
        self.env.reset()

        # Reset the learner to reset any trajectory potentially stored in the learner
        # IMPORTANT: We must reset the learner AFTER resetting the environment so that the first reward stored in the learner is the reward associated to the reset state.
        # (as opposed to the reward of the state that was visited at the end of a potential previous experiment!)
        learner.reset(reset_episode=True, reset_value_functions=True, reset_average_reward=True)

        t = 0  # Step counter: the first step is 1, as t represents the time at which the Markov chain transitions to the NEXT state. See more details at the @note at the beginning of the file.
        t_episode = -1  # Step counter within episode: the first step is 0, as t_episode indexes the step BEFORE transition so that we can write S(0), A(0), R(1), S(1), A(1), ...
        done_episode = self.env.getState() in self.env.getTerminalStates()
        while t < max_time_steps:
            t += 1
            t_episode += 1

            state = self.env.getState()

            if done_episode:
                # We have reached a terminal state
                # => Reset the environment and the trajectories stored in the learner

                t_episode = -1
                action = 0
                next_state = self.env.reset()
                reward = self.env.getReward(next_state)
                done_episode = next_state in self.env.getTerminalStates()

                # TEMPORARY (2024/05/14): Needed only because of the EPISODIC view of the average reward in its iterative update formula in Learner.update_average_reward()
                # Partially reset the learner (only trajectories are reset). See why we need to do this where we do the same thing in _run_single_continuing_task()
                learner.reset(reset_episode=False, reset_value_functions=False, reset_average_reward=False)

                # See the reasons why we set these parameters where we do the same thing in _run_single_continuing_task()
                info = {'update_trajectory': False,
                        'update_counts': False}
                # TEMPORARY (2024/05/14): Needed only because of the EPISODIC view of the average reward
            else:
                action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
                next_state, reward, done_episode, info = self.env.step(action)

            # TEMPORARY (2024/05/14): Needed only because of the EPISODIC view of the average reward in its iterative update formula in Learner.update_average_reward()
            # Check end of simulation before learning so that we make the learner do what it usually does at the end of an episode
            # (e.g. learn the average reward and update trajectory information over all episodes)
            if t >= max_time_steps:
                done_episode = True
                if self.debug:
                    print("[run_exploration, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, learner.episode))
            # TEMPORARY (2024/05/14): Needed only because of the EPISODIC view of the average reward

            # Learn (and update the trajectory stored in the learner)
            learner.learn(t_episode, state, action, next_state, reward, done_episode, info)
            if not self.env.isStateContinuous():
                if state in self.env.getTerminalStates():
                    # Copy to all possible actions the Q-value just learned for the taken (dummy) action used to reset the environment when reaching a terminal state
                    self._copy_action_values_for_terminal_state(learner.getQ(), state, action)
                    self._copy_action_values_for_terminal_state(learner.getA(), state, action)

            if show_messages(verbose, verbose_period, t):
                print(f"t: {t}, t in episode: {t_episode}, s={state}, a={action} -> ns={next_state}, r={reward}, " +
                      "avg. reward in episode = {:.4f}, avg. reward = {:.4f}".format(learner._average_reward_in_episode, learner.getAverageReward()) +
                      "\nV={:.3f}, Q={:.3f}".format(self._get_state_value(learner, state), self._get_action_value(learner, state, action)))

        return learner, t

    @measure_exec_time
    def _run_simulation_fv(self, t_learn, envs, absorption_set,
                           start_set: set=None,
                           max_time_steps=None,
                           max_time_steps_for_absorbed_particles_check=+np.Inf, min_prop_absorbed_particles=1.0, stopping_criterion_fv=StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES,
                           dist_proba_for_start_state: dict=None,
                           start_state_selection_case: str="",
                           soft_killing: bool=False,
                           dict_proba_killing: dict=None,
                           update_absorption_set_with_fv_visits: bool=False,
                           reward_shaping=False,
                           expected_absorption_time=None, expected_exit_time=None,
                           estimated_average_reward=None,
                           use_fixed_average_reward=False,
                           keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=True,
                           epsilon_random_action=0.0,
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
            The policy learning time step (starting at 0) to which the simulation will contribute.
            Only used for informative purposes on where we are at when learning the policy or to decide whether to show information in the log.

        envs: list
            List of environments used to run the FV process.

        absorption_set: set
            Set with the entrance states to the set of uninteresting states A.
            This set is needed to measure the killing times used to estimate the survival probability P(T > t).

        start_set: (opt) set
            Set of states where the FV particles can be placed at the beginning of the simulation, which are selected uniformly at random.
            This is normally the outer boundary of the absorption set A, i.e. the entrance set of to the complement of A.
            When None, it is assumed that the dictionary parameter `dist_proba_for_start_state` is not None so that start_set is defined
            as the keys of such dictionary.
            default: None

        max_time_steps: (opt) int
            See description in the documentation for _run_fv().
            default: None, in which case 100*N is used, where N is the number of particles in the FV system

        max_time_steps_for_absorbed_particles_check: (opt) int
            See description in the documentation for _run_fv().
            default: +np.Inf

        min_prop_absorbed_particles: (opt) float in [0, 1]
            See description in the documentation for _run_fv().
            default: 1.0

        stopping_criterion_fv: (opt) StoppingCriterion
            See description in the documentation for _run_fv().
            default: StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES

        dist_proba_for_start_state: (opt) dict
            Probability distribution to use for the selection of the start state of each FV particle.
            According to theory, the start state should be selected from the activation set and should be selected
            following the stationary EXIT state distribution of the absorption set A
            (i.e. the stationary ENTRY state distribution to the complement of A).
            If this is not available, a reasonable approximation (that should beat the uniformly random selection of the start state)
            is the stationary distribution of the outer boundary of A (as opposed to the stationary *exit* distribution).

            Two situations may occur for the selection of the start state of the FV particles:
            a) If `start_set` is given, the set of start states to select from is given by `start_set`
            b) If `start_set` is None, the set of start states to select from is given by the keys of the `dist_proba_for_start_state` dictionary.

            Note that `start_set` and `dist_proba_for_start_state` CANNOT be both None, as the set of start states for the FV particles must be defined
            in order to select the start states!

            default: None, in which case a uniform distribution on `start_set` is used

        soft_killing: (opt) bool
            Whether particles are killed following a probability distribution defined for each state.
            default: False

        dict_proba_killing: (opt) dict
            Dictionary containing the killing probability for each state as long as it has killing probability > 0.
            States where killing should not happen do NOT need to show up here.
            This is used when using soft killing to kill particles, i.e. where a particle is killed at certain states when a killing clock is signalled.
            default: None

        update_absorption_set_with_fv_visits: (opt) bool
            Whether the absorption set should be updated based on the visit frequency by the FV simulation.
            default: False

        reward_shaping: (opt) bool
            Whether the environment has been affected by reward shaping.
            This is important only when update_absorption_set_with_fv_visits = True, so that states that were potentially affected by reward shaping
            (i.e. that were assigned a non-zero reward while there original reward is 0 --e.g. EXIT states from A) can be selected to enlarge A
            (because only states with no reward are accepted as part of the enlargement of A).
            default: False

        expected_absorption_time: (opt) positive float
            Expected reabsorption time E(T_A) used to estimate the stationary state probability distribution.
            When given, this value is also used by the iterative update of the FV-based average reward, which is updated at every new survival time
            observed for an FV particle that has never been absorbed before.
            If None, the expected reabsorption time E(T_A) is estimated as the sum of `expected_exit_time`
            (which should be given when this parameter is None) and the average of the first N survival times of the respective FV particles.
            In addition, the FV-based estimate of the average reward is carried out at the end of the simulation, once all observations
            needed to estimate P(T>t) and Phi(t,x) are available.
            default: None

        expected_exit_time: (opt) positive float
            Expected exit time from the absorption set, which is needed when expected_absorption_time = None, as this
            value will be used to estimate it as expected_exit_time + "expected killing/survival time" computed as the average
            of the first N survival times observed for the respective N FV particles from where P(T>t) (df_proba_surv) is estimated.
            This value should be provided when expected_absorption_time is None. If the latter is provided, the value is ignored.
            default: None

        estimated_average_reward: (opt) None
            An existing estimation of the average reward, used as follows:
            - when stopping_criterion_fv = StoppingCriterion.MAX_STEPS or MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES,
            it is the average reward that is used as fixed correction value for the computation of the value functions.
            Note that, under these two criteria, the average reward is NOT iteratively updated during the FV simulation process
            because it is actually estimated at the END of the FV simulation (by the caller method) using a parametric estimation of
            the survival probability distribution, P(T>t), which is based on having observed ALL the survival times (used in the estimation of E(T)).
            If the value is `None` or missing, its value is set to zero, with a warning.
            - when stopping_criterion_fv = StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES or
            StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN,
            this average reward is used as initial value for the iterative FV-based estimation of the average reward,
            where the average reward is updated every time a new survival time (used in the estimation of P(T>t), normally the first survival time
            of each particle) is observed. When `None`, the FV-based average reward is estimated from scratch, i.e. from zero.
            Usually, this estimated average reward comes from the single Markov chain simulation that is used to estimate the expected reabsorption time, E(T_A).
            default: None

        use_fixed_average_reward: (opt) bool
            Whether to use the average reward given in `estimated_average_reward` (when not None or missing) as a fixed correction of the value functions
            estimated by the learner during this simulation.
            If True is given but the value of `estimated_average_reward` is missing, an error is raised.
            default: False

        keep_fv_estimation_of_average_reward_and_stationary_probability_consistent: (opt) bool
            Whether to guarantee that the average reward estimated by the FV estimator is consistent with the estimated stationary probability distribution
            outside A, meaning that the following equality is satisfied:
                "average reward estimated by FV" = \sum_{x outside A} p(x) r(x)
            Set this parameter to False if we are only interested in using the average reward signal for policy learning,
            as opposed to estimation purposes which require that the above consistency be satisfied.
            default: True

        epsilon_random_action: (opt) float in [0, 1]
            Probability to take a random action instead of choosing an action dictated by the policy.
            This is useful to guarantee ergodicity when policies are deterministic or have some actions with zero probability.
            default: 0.0

        seed: (opt) int
            Seed to use in the simulations as base seed (then the seed for each simulation is changed from this base seed).
            default: None, in which case a random seed is generated.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        plot: (opt) bool
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: (opt) str
            Name of the colormap to use in the generation of the animated plots showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.
            default: seismic, a colormap that ranges from blue to red, where the middle is white

        pause: (opt) float
            Number of seconds to wait before updating the plot that shows the evolution of the value function estimates.
            It should be positive if we want the plot to be updated every time the value function estimate is updated.
            default: 0.1

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
        - absorption_set: the updated absorption set based on the visit frequency of the FV particle system under VALID transitions of the original Markov process.
        This can be used to expand the absorption set by including frequently visited states by the FV particles that yield no reward.
        NOTE: the absorption set is updated ONLY when enough number of absorption cycles have been observed (say 5), o.w. it would be risky to enlarge the
        absorption set because it may well be the case that no more cycles are observed under the current policy.
        - less_frequently_visited_set: set of states visited by the FV particles (under VALID transitions of the original Markov process) that are not part of
        the most visited set by the FV particles and that have NOT been added to the `absorption_set`. This might be useful as backup information
        in extreme cases where the choice of the start states of the FV particles is not clear due to lacking to observe EXIT states from the absorption set.
        """
        #------------------------------- Auxiliary functions ----------------------------------#
        def reactivate_particle_internal(idx_particle):
            "Internal function that reactivates a particle"
            state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                # We can add a `True` condition above in order to show the following useful piece of information of the proportion of particles at the state of interest x is useful for tracking the stabilization of Phi(t,x) around the QSD
                flags_particle_at_terminal_state = [1 if envs[p].getIndexFromState(envs[p].getState()) in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of {})".format(np.mean(flags_particle_at_terminal_state)*100, np.sum(flags_particle_at_terminal_state), len(envs)))

            idx_reactivate = reactivate_particle(envs, idx_particle, 0, reactivation_number=None)
                ## (2023/01/05) the third parameter is dummy when we do NOT use method = ReactivateMethod.VALUE_FUNCTION to reactivate the particle inside function reactivate_particle().
                ## (2024/01/28) Use reactivation_number=reactivation_number to indicate that we want to use method = ReactivateMethod.ROBINS.
            new_state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                print("*** t={}: Particle #{} ABSORBED at state={} and REACTIVATED to particle #{} at state {}".format(t, idx_particle, state, idx_reactivate, new_state))

            return new_state

        def update_average_reward(learner, survival_times, expected_absorption_time, estimated_average_reward_at_start_of_fv_process=None):
            """
            Updates the average reward stored in the given learner ITERATIVELY, based on the newly observed survival time,
            which is assumed to be the FIRST absorption time of each particle, so as to guarantee survival times that are observed in increasing order,
            a condition that is required by the average reward update procedure.

            Arguments:
            learner: LeaFV
                This is the FV learner where all the iterative estimates needed to update the average reward are stored (e.g. the FV integral, Phi, etc.).

            survival_times: list
                List of survival times observed so far, where the newly observed survival time is the last element of the list.
                This last survival time is used to update the FV integral which in turn is used to update the average reward,
                using the value of the expected reabsorption time.

            expected_absorption_time: positive float
                Estimate of the expected reabsorption time E(T_A) that is used to estimate the stationary probabilities that are used to update
                the estimate of the average reward.

            estimated_average_reward_at_start_of_fv_process: (opt) float
                Initial estimate of the average reward that is used as initial hint of the average reward value that is currently being estimated.
                Normally this would be an estimate of the average reward from a previous FV simulation, e.g. carried out under the policy estimated
                at the previous policy learning step.
                This value is normally kept fixed as the current FV simulation carries on, and is used to update the FV estimate of the average reward as:
                    avgR(n) = avgR(n-1) + (updated_fv_integral(n) / E(T_A) - avgR(n)) * n / (N + n)
                where n is the number of survival times observed so far and used in the calculation of the FV integral,
                and updated_fv_integral(n) is the result of the iterative update of the FV integral after observing the n-th survival time.
                When None **or when the contribution to the integral update comes from the last possible observed survival time (i.e. the N-th survival time)**,
                the average reward is simply set to:
                    avgR(n) = updated_fv_integral(n) / E(T_A)
                Note that, when the update is computed on the last possible observed survival time (i.e. the N-the survival time), we finally have a complete
                picture of the average reward estimated by the FV simulation, so there is no need to weigh in an initially computed average reward
                --as the weighing is done simply to get a better estimate for the average reward (i.e. a possibly non-zero estimate) WHILE the FV simulation
                is still going on... but this should no longer be done at the END of the FV simulation, because we want to base the average reward estimation
                SOLELY on the FV simulation when we effectively arrive to the end of the FV simulation.
                In addition and MOST IMPORTANTLY, when the FV simulation finished, the average reward must be consistent with the estimated stationary
                probabilities and the reward landscape, i.e. the average reward should satisfy its definition as expected reward under stationarity, namely:
                    avgR = sum{x} p(x)*r(x)
                where p(x) is the estimated stationary probability for state x and r(x) is the reward received when visiting state x.
                default: None

            Return: float
            The updated average reward value following the newly observed survival time.
            Note that this may NOT be the average reward stored in the learner when parameter estimated_average_reward_at_start_of_fv_process is not None,
            as in that case the average reward stored in the learner is a weighted average of the value given in that parameter and the updated average reward,
            as described above.
            """
            assert expected_absorption_time is not None and expected_absorption_time > 0, f"The expected reabsorption time must be given and be positive ({expected_absorption_time})"

            # Initialize the dictionary with the estimated stationary probabilities that is used to compute the updated average reward,
            # based on the updated FV integral, which in turn is based on the updated Phi(t, x) function in the latest inter-survival interval [t_surv_prev, t_surv).
            probas_stationary = dict.fromkeys(learner.getIntegral(), 0.0)

            # Latest observed survival time which is used to update the FV integral
            t_surv = survival_times[-1]
            learner.update_integral(t_surv, fixed_sample_size=True)
            for _state, _integral_for_state in learner.getIntegral().items():
                probas_stationary[_state] = _integral_for_state / learner.getNumParticles() / expected_absorption_time
            updated_average_reward = compute_expected_reward(self.env, probas_stationary, require_all_states_with_reward_present_in_dictionary=False)
                ## NOTE: We do NOT require that all states with reward be present in the dictionary containing the stationary probability
                ## because some of those states may be present in the absorption set A, which should NOT be considered in the estimation of
                ## the expected reward outside A.
            n_survival_times_observed_so_far = len(survival_times) - 1  # -1 because the first value in the survival_times list is a dummy survival time of 0
            if  estimated_average_reward_at_start_of_fv_process is None or \
                keep_fv_estimation_of_average_reward_and_stationary_probability_consistent and n_survival_times_observed_so_far == N:
                    # (2025/06/20) Use the following additional condition, if we want to use the "pure" FV average reward value when it is larger than the initial one
                    # (i.e. typically the average reward from an initial exploration or the one stored in the learner as obtained from previous policy learning steps)
                    #or np.abs(updated_average_reward) > np.abs(estimated_average_reward_at_start_of_fv_process):
                # => The updated average reward computed above (as FV_integral_on_reward_values / E(T_A)) is directly the estimate of the average reward
                # we store in the learner to be used as correction for the value functions.
                # Notes:
                # a) This makes the average reward CONSISTENT with the stationary probability distribution outside A,
                # meaning that if we compute \sum_{x outside A} p(x) r(x) we must get the average reward stored in the learner.
                # In fact the stationary probability distribution outside A is computed as FV_integral / E(T_A),
                # i.e. in the same way as the average reward but using the indicator function I(x) instead of the reward function r(x) in the integrand.
                # b) If the average reward computed at the end of the FV simulation is 0.0, then that would be the value
                # stored as average reward in the learner, even if a non-zero initial estimate of the average reward
                # was passed as parameter estimated_average_reward_at_start_of_fv_process...
                # However, this is unlikely because the FV simulation is expected to estimate a non-zero average reward when the initial E(T_A) excursion
                # estimates a non-zero average reward. What is more likely though is the situation where the average reward estimated by the FV simulation
                # is zero but the average reward estimated at the PREVIOUS policy learning step is not zero... this MIGHT make sense (although not guaranteed)
                # in the case when the current average reward is computed on a different policy than the policy at the previous learning step.
                learner.setAverageReward(updated_average_reward)
            else:
                # We have a START value for the average reward (possibly provided from the FV estimate under a previously learned policy)
                # AND we don't care about keeping a consistency between the estimated average reward and the estimated stationary probability distribution,
                # i.e. we care only on having a "large enough" value of the average reward that may provide sensible information for policy learning.
                # => Recursively update the average reward using the newly updated FV-based average reward just computed,
                # Note that the weight given to the current estimate of the average reward is equal to the fraction of survival times observed so far
                # out of ("number of particles" + "number of survival times observed so far").
                # By doing this, we are assuming that the previous estimate of the average reward (stored in the learner) was done on as many survival times as
                # the number of particles in the FV system, which is a reasonable assumption, as normally such previous estimate was computed by this same iterative
                # procedure done here.
                # (recall the k-step update of an average X(n) as X(n+k) = X(n) + (X(n, n+k) - X(n)) * k / (n+k), where X(n, n+k) is the average observed between
                # observations n and n+k), and this means that the weight given to the current average X(n) is n/(n+k) and the weight given to the latest observed
                # average, X(n, n+k) is k/(n+k))

                #if updated_average_reward != 0.0:  # <- (2025/06/20) Use this if we want to avoid using the FV average reward when it is estimated as zero (in order to avoid reducing the potentially positive average reward estimated from the initial exploration and/or already stored in the learner)
                learner.setAverageReward(estimated_average_reward_at_start_of_fv_process + (updated_average_reward - estimated_average_reward_at_start_of_fv_process) * n_survival_times_observed_so_far / (learner.getNumParticles() + n_survival_times_observed_so_far))

            return updated_average_reward
        #------------------------------- Auxiliary functions ----------------------------------#


        #----------------------------- Parse input parameters ---------------------------------#
        # Absorption set
        if absorption_set is None or not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a given and must be a set: {}".format(absorption_set))
        if soft_killing and dict_proba_killing is None and not isinstance(dict_proba_killing, dict):
            raise ValueError("Parameter `dict_proba_killing` must be a dictionary when `soft_killing=True`: {}".format(dict_proba_killing))

        # Start set and probability distribution for the start state
        if start_set is None:
            if dist_proba_for_start_state is None:
                raise ValueError("Parameters `start_set` and `dist_proba_for_start_state` cannot be both None")
            if not isinstance(dist_proba_for_start_state, dict):
                raise ValueError(f"Parameter `dist_proba_for_start_state` must be a dictionary:\n{dist_proba_for_start_state}")
            print(f"Distribution of start states for FV particles:")
            _cumprob = 0.0
            for state in sorted(dist_proba_for_start_state, key=lambda x: -dist_proba_for_start_state[x]):  # `key=` requests sorting the values in the dictionary in decreasing order
                _cumprob += dist_proba_for_start_state[state]
                print("{}: {:.2f}%, [{:.1f}%]".format(state, dist_proba_for_start_state[state]*100, _cumprob*100))
            start_set = set(dist_proba_for_start_state.keys())
        # The set of start states is given directly by the user, as opposed to its distribution
        if not isinstance(start_set, set):
            raise ValueError("The set of start states for the FV particle system (`start_set`) must be a set: {}".format(start_set))
        if len(start_set) == 0:
            raise ValueError("The set of start states for the FV particle system (`start_set`) must have at least one element")
        if not soft_killing and len(start_set.intersection(absorption_set)) > 0:
            raise ValueError(f"The start set must NOT contain any state in the absorption set. States present in the absorption set:\n{start_set.intersection(absorption_set)}")

        if  stopping_criterion_fv in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES] \
            and estimated_average_reward is None:
            # Set the estimated average reward to 0.0 when the stopping criterion requires to always reach MAX_TIME_STEPS but the given estimated average reward is missing
            # This is necessary because in those cases the average reward is NOT estimated iteratively as the FV simulation progresses, but a pre-calculated fixed value is used.
            warnings.warn(f"Under the {stopping_criterion_fv.name} simulation stopping criterion, the estimated long-run expected reward used to correct the value functions should not be `None` "
                          f"(because such expected reward is NOT estimated iteratively as the FV simulation progresses). However, `None` has been given, and the value zero will be used instead.")
            estimated_average_reward = 0.0
            ## NOTE: This average reward value will be set in the learner below, by a call to learner.setAverageReward()

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")

        if use_fixed_average_reward and (estimated_average_reward is None or np.isnan(estimated_average_reward)):
            # Do NOT use a fixed average reward if the given average reward value is None or missing
            # Note that we would enter this condition ONLY when the stopping criterion are NOT MAX_TIME_STEPS or MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES
            # because in those cases, a potential missing value of estimated_average_reward has already been taken care of by defining its value as 0.0 above.
            use_fixed_average_reward = False
            raise ValueError("The user requested to use a FIXED value of the average reward as correction of the value functions but the given average reward in `estimated_average_reward` is None or missing.")
        #----------------------------- Parse input parameters ---------------------------------#

        N = len(envs)
        if max_time_steps is None:
            max_time_steps = N*100          # N * "MAX # transitions allowed on average for each particle"
        policy = self.agent.getPolicy()     # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Set seeds of:
        # - torch --> responsible for selecting the action when the policy is modeled via a neural network.
        # - the policy's environment --> just in case this environment does NOT have the same memory address as self.env
        #   (e.g. when policies are deepcopied from the original policy in order to compare different learning methods)
        # - numpy [NOT SET, see below] --> responsible for deciding whether a (completely) random action is chosen (when epsilon_random_action > 0).
        #   numpy is used to both draw a random number to decide whether to choose a random action (without following the policy)
        #   and then to actually choose that random action, if this ends up being the case.
        #   NOTE HOWEVER: We do NOT set the numpy random seed because for FV this has already been set at the very beginning of the process,
        #   by method _estimate_value_functions_and_expected_reward_fv(). Since this is not the case in the single Markov explorations
        #   run by the run_exploration*() methods, the numpy random seed is set in those methods.
        #   Another reason for not setting the numpy random seed here is that the expected results of FV unit tests would change and I don't want to update them now.
        if seed is not None:
            torch.manual_seed(seed)
            policy.env.seed(seed)
            #np.random.seed(seed)   # Not set because of the reasons indicated in the above comments

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # (UNLESS reward shaping was performed during the initial exploration of the environment to promote a policy that takes the agent towards the boundary of A,
        # in which case we should discard any information about the value functions that came from that phase, because they correspond to a modified environment
        # --although that this is the best strategy is not yet clear).
        # The average reward is reset when no initially estimated average reward is given. If such initially estimated average reward is given,
        # it means that it should be used as initial estimate of the average reward during further learning. Otherwise, when it is not given,
        # it means that we start the average reward learning process from scratch (i.e. from an average reward initially estimated as zero).
        # IMPORTANT: The reset of the auxiliary information (e.g. number of particles, absorption set, FV start states distribution, etc.) is NOT done
        # because we might need auxiliary information stored in the FV learner for the following policy learning step, namely the expected absorption time E(T_A)
        # estimated during the current policy learning step. This value is used as replacement of the expected absorption time
        # estimated by the FOLLOWING policy learning step in case no cycles are observed during the single Markov chain simulation; if we reset this value here,
        # (i.e. at the first policy learning step having t_learn=0), its value would be np.nan and there is a risk that an error will be triggered
        # if no cycles are observed at the next policy learning step.
        # Another important aspect of this reset step (which always happens, regardless of parameter values) is the reset of the learning rates (alpha) of all states and actions,
        # so that we start the FV-based learning with full strength).
        learner.reset(reset_episode=True, reset_value_functions=reward_shaping, reset_average_reward=estimated_average_reward is None, reset_auxiliary_info=False)
        if reward_shaping:
            # Reset the value functions INSIDE the absorption set A, so that we keep whatever was learned already outside A!
            for s in absorption_set:
                learner.getV().setValue(s, 0.0)
                for a in range(self.env.getNumActions()):
                    learner.getQ().setValue(s, a, 0.0)
                    learner.getA().setValue(s, a, 0.0)

        # Store the estimated average reward passed by the user as the average reward value of the learner,
        # so that it can be used as correction value when learning the differential value functions (which is precisely the average reward)
        # This value is updated iteratively as the FV simulation proceeds, so the correction is updated during the simulation process,
        # UNLESS parameter use_fixed_average_reward = True, in which case this `estimated_average_reward` is ALWAYS used as correction, as a fixed value.
        if estimated_average_reward is not None and not np.isnan(estimated_average_reward):
            learner.setAverageReward(estimated_average_reward)
        else:
            # I write here an assertion and not an IF condition with information to the user because the condition that would make the assertion fail has already been taken care of
            # in the "Parse input parameters" section. We write the assertion here just as a double check.
            assert stopping_criterion_fv not in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES], \
                "The estimated_average_reward value cannot be missing when the stopping criterion is MAX_TIME_STEPS or MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES" \
                " because its value is used as (fixed) correction in the estimation of the value functions."

        # Make copies of the learner if the learner is TD(lambda) instead of TD(0)
        # so that we can keep the appropriate history for EACH PARTICLE on which value functions should be updated using TD(lambda),
        # namely, the trajectory starting from the reactivation state, NOT before that because cells are not linked before and after absorption by the original Markov process.
        # NOTES:
        # - The original `learner` object is still used to learn the average reward and Phi, as they require pooling information from ALL particles.
        # - All learners share the SAME value functions V and Q because their information needs to be shared and because the environment has only ONE V and ONE Q to estimate!
        # - When learning with each particle's learner separately, the COMMON average reward is used as correction.
        # - We still need to store the trajectory in the original `learner` object because this is used to decide whether new states start being part of A because of high frequency
        # visit during the FV simulation (see `if update_absorption_set_with_fv_visits` block below).
        is_learner_td_lambda = learner.lmbda > 0
        if is_learner_td_lambda:
            learners = [copy.deepcopy(learner) for i in range(N)]

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        if dist_proba_for_start_state is None or len(dist_proba_for_start_state) == 0:
            print(f"The distribution for the start state selection of the FV particles is None. A UNIFORM distribution on the states in `start_set` will be used:\n{start_set}")
        else:
            print(f"Probability distribution used to select the start states of the FV particles:\n{dist_proba_for_start_state}")
        if len(start_set) == 1:
            # When there is only one set in the start set, there is no need to use a probability distribution and make the process waste time in choosing a state randomly
            # in a singleton!
            # In addition, if we run a random choice selection of the start state explicitly specifying the probability distribution,
            # even if this probability distribution is on a single state and is equal to p=1.0, the random selector draws a random number,
            # whereas this seems NOT to be the case if the probability distribution of np.random.choice() is not specified (via parameter `p`).
            # (this was proved by running the following two lines and observing that the results of the second random draw are different from the first one:
            #   np.random.seed(1713); print(np.random.choice(list({3}))); np.random.choice(sorted({3, 2, 1}), 5, p=[0.5, 0.3, 0.2])
            #   np.random.seed(1713); print(np.random.choice(list({3})), p=[1.0]); np.random.choice(sorted({3, 2, 1}), 5, p=[0.5, 0.3, 0.2])
            # Note that the results are also different when the set of values to select from is *more than one* and `p` is defined as the uniform distribution;
            # i.e. when we define `p=` and when we NOT pass `p=` explicitly to the first call to np.random.choice() in each of the two lines above
            # (which means that the value is chosen uniformly at random), results of such random selection are different!
            # )
            # (2024/04/14) I found this out after noting that the number of events observed during the FV simulation changed when I started using a probability distribution
            # to select the start state (defined in `dist_proba_for_start_state`) on a singleton set (the context is a 1D gridworld),
            # via the call to the choose_state_from_set() function below, which precisely calls np.random.choice() passing a value for the `p=` parameter
            # when the `dist_proba_start_state` parameter is NOT None.
            dist_proba_for_start_state = None

        # Choose the start state from the activation set or a subset of it, if not all those states are present in the dictionary of the start state distribution
        # --because e.g. they were not observed during the E(T_A) excursion used for the stationary exit state distribution.
        # First parse the start set parameter
        _start_set_parsed = start_set.copy()
        if dist_proba_for_start_state is not None:
            if len(dist_proba_for_start_state) == 0:
                warnings.warn(f"[_run_simulation_fv] The dictionary containing the start state distribution to use `dist_proba_for_start_state` is empty."
                              f"\nThe start states will be selected UNIFORMLY AT RANDOM (but this may not be what is wished).")
                dist_proba_for_start_state = None
            elif not set(dist_proba_for_start_state.keys()).issubset(start_set):
                # TODO: (2024/12/23) Update this logic (defining the distribution for the start state selection) so that when a start set is given and the distribution of start states is not None, the states in the given set are selected using the probability distribution stored in the start state distribution dictionary (dist_proba_start_state) READJUSTED to the intersecting states with `start_set`, instead of selecting them uniformly at random (which is the case now when setting dist_proba_for_start_state to None).
                warnings.warn(f"[_run_simulation_fv] The set of start states given in `start_set` ({start_set}) "
                              f"is NOT a subset of the keys present in the dictionary containing the start state distribution to use `dist_proba_for_start_state`:\n{dist_proba_for_start_state}"
                              f"\nThe start states will be selected UNIFORMLY AT RANDOM (but this may not be what is wished).")
                dist_proba_for_start_state = None
            else:
                # Define the set of start states as the keys of the given distribution, which define precisely which states the start state should be chosen from
                _start_set_parsed = set(dist_proba_for_start_state.keys())
        for i, env in enumerate(envs):
            # Environment seed
            # IMPORTANT: (2024/05/07) Setting this seed is ONLY relevant when the environment has some stochastic component --e.g. wind in a gridworld.
            # HOWEVER, the seed is NOT used to choose the action taken by the agent! In fact, the seed for the agent's action is the seed that was used when defining
            # the random number generator of the environment stored in the `policy` object. This seed is set above.
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)
            start_state = choose_state_from_set(_start_set_parsed, dist_proba_for_start_state)
            env.setState(start_state)

            if is_learner_td_lambda:
                # Share information about the learning process that should be common among particles, namely:
                # - value functions: all particles are exploring the environment in parallel to learn the same value functions!
                # - visit counts: since all particles are learning the same value functions, their strength (learning rate) when updating the value functions should be adapted
                #   based on what the other particles learn, otherwise the learning rate would NOT be adjusted by the visit counts observed by the system, and could be too large,
                #   possibly generating too much oscillation in the value function estimates.
                # - learning rates: learning rates may not need to be shared across particles because they are adjusted using visit counts at each learning rate update step
                #   from the original learning rate value. However, sharing them helps monitor their values more easily, e.g. when constructing plots.
                learner.share_value_functions(learners[i])
                learner.share_visit_counts(learners[i])
                learner.share_learning_rates(learners[i])

        # Event times: the first event time is 0
        # This implies t=0 is considered the time at which each particle is positioned to their start state.
        event_times = [0]

        # Phi(t, x): Empirical probability of the states of interest (x)
        # at each time t when a variation in Phi(t, x) is observed.
        # TODO: (2025/01/19) We should initialize Phi to the location of the FV particles and then update it on every new state that is visited by the FV particles, so that all states can contribute to the average reward!
        # The above to-do is specially important when we do reward shaping to promote the exploration of the states near the exit states from A, as done since yesterday 18-Jan-2025
        # Currently, the reward shaping strategy is tested by (TEMPORARILY) assigning the average reward estimated during the initial exploration as estimated expected reward.
        learner.dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getStatesOfInterest())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = [0]    # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        idx_reactivate = None   # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
        # List that keeps track of whether each particle has been absorbed once so that we can end the simulation when all particles have been absorbed at least once
        # (only taken into account when the stopping criterion is different from MAX_TIME_STEPS)
        has_particle_been_absorbed_once = [False]*N
        n_particles_absorbed_once = 0

        # Plotting setup
        if plot:
            #-- Setup the concepts and figures to plot (e.g. that will be updated at every absorption event)
            # Setup the figures that will be updated at every verbose_period
            num_colors_in_colormap = N # Use the following if we plan to update the plot of V(s) at verbose_period (instead of at every absorption event) #None if max_time_steps is None or max_time_steps == +np.Inf else max_time_steps
            fig_V, fig_V2, fig_C, fig_RMSE_state, colors_V, _ = self._setup_plots(t_learn=t_learn, colormap=colormap, lut=num_colors_in_colormap, setup_policy_plot=False)

            # Setup the axes to use for each plotted object, indexed by the name of the plotted object (e.g. "average_reward")
            dict_axes = dict({'average_reward': plt.figure().subplots(1, 1)})
            # Initialize the average reward plot with the current estimate of the average reward stored in the learner
            # Note that we add two line objects, one that will plot the updated average reward coming solely from the FV excursion (green line),
            # and the other line will plot the updated average reward stored in the learner, which may have a contribution from the original average reward stored in the learner (red line).
            # Note that these line properties will be preserved when updating the plot by extracting the information from the line objects returned by plot() below (see plotting.update_plots())
            dict_lines = dict({'average_reward': [dict_axes['average_reward'].plot(0, learner.getAverageReward(), '.-', color="green", linestyle="solid")[0],
                                                  dict_axes['average_reward'].plot(0, learner.getAverageReward(), '.-', color="red", linestyle="dashed")[0]]})
            # Set the maximum X axis value if the number simulation steps to run in advance is known
            dict_axes['average_reward'].set_xlim((None, max_time_steps))
            dict_axes['average_reward'].set_xlabel("Survival time contributing to P(T>t)")
            dict_axes['average_reward'].set_ylabel("Estimated average reward")
            dict_axes['average_reward'].legend(["FV-based average reward", "Weighted initial + FV-based average reward"])
            plt.suptitle(f"[_run_simulation_fv, Learning step {t_learn+1}]")

            # Reposition and resize the figure to avoid overlapping with the other figures
            # Ref: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
            plt.figure(dict_axes['average_reward'].get_figure().number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry( WINDOW_TOP_LEFT_HORIZONTAL + WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS,
                                        WINDOW_TOP_LEFT_VERTICAL + WINDOW_HEIGHT + 3*SPACE_BETWEEN_WINDOWS, # `3*` because we need to leave space for the WINDOW's title
                                        WINDOW_WIDTH,
                                        WINDOW_HEIGHT)

        if DEBUG_ESTIMATORS: #True or DEBUG_ESTIMATORS:
            if self.env.isStateContinuous():
                V = np.repeat(0.0, self.env.getNumStates())
                Q = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
                A = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
            else:
                V = learner.getV().getValues()
                Q = learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())
                A = learner.getA().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(V))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(Q))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(A))

        idx_particle = -1
        done = False
        t = 0   # Learning step counter: t represents the time at which a particle transitions to the NEXT state. See more details at the @note at the beginning of the file.
        # Initial learning rate for Phi, which is the gamma* defined in Fraiman et al.
        alpha0 = +np.Inf    # Setting this initial learning rate to 1 gives the same weight to all historic (k < t) and current (k = t) Phi values
                            # setting it equal to > 1 gives more weight to most recent estimates of Phi,
                            # setting it equal to < 1 gives more weight to older estimates of Phi.
        n_consecutive_steps_at_same_system_state = 0    # Counter of what the variable name indicates that is used to check whether the system gets stuck (see explanation and example below, where this variable is updated)
        rewards_after_reactivation = deque([])  # List of rewards observed after each reactivation, which are NOT kept track of by the learner (because no value function learning should take place after reactivation because the transition does not correspond to a valid transition of the original Markov chain
        info = dict()       # We need this variable to be defined when calling learner.learn() for the first time if the first particle picked for moving has started at a terminal state

        if plot:  #False:
            # Initialize the plot of the distribution of the FV particles in the environment
            ax_dist_fv = plt.figure().subplots(1, 1)
            _state_counts = np.zeros(self.env.getNumStates())
            _dist_fv_particles = pd.Series([env.getState() for env in envs]).value_counts(sort=False)
            _state_counts[_dist_fv_particles.index] = _dist_fv_particles
            self.env.plot_values(_state_counts, ax=ax_dist_fv, cmap="Blues", vmin=0, vmax=N)
            # Add the absorption set as crosses
            self.env.plot_points(list(absorption_set), ax=ax_dist_fv, color="red", markersize=5, style="x")
            ax_dist_fv.set_title(f"Learning step t_learn = {t_learn + 1}"
                                 f"\nDist. of FV particles (N={N}) at t = {t} of {max_time_steps}")
            plt.pause(0.01)
            plt.draw()

        while not done:
            t += 1

            # We define a reactivation number, an integer between 0 and N-2 which is used to deterministically choose
            # the reactivation particle (if reactivation is via ReactivateMethod.ROBINS), in order to save time by not having to generate a uniform random number.
            # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle can be reactivated into.
            reactivation_number = t % (N - 1)

            event_times += [t]

            # Select the particle to move uniformly at random
            idx_particle = np.random.choice(N)  # If choosing them in order, use: `(idx_particle + 1) % N`

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
            if self.env.getIndexFromState(state) in self.env.getTerminalStates():
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
                if learner.getLearningTask() == LearningTask.CONTINUING:
                    if is_learner_td_lambda:
                        # Use TD(lambda) on each particle separately, BUT using the COMMONLY estimated average reward (since we need all particles to do so)
                        info['average_reward'] = estimated_average_reward if use_fixed_average_reward else learner.getAverageReward()
                        self.learn_terminal_state_values(learners[idx_particle], t, state, next_state, reward, info, done_episode=next_state in self.env.getTerminalStates())
                        # Update the trajectory and state counts stored in the base learner so that we can use them when analyzing whether the absorption set A should be increased
                        # based on the FV visits. Note that the third argument 0 is the default (anchor) action that is considered to be taken to transition from a terminal state
                        # (see the learn_terminal_state_values() method for more details).
                        learner._states += [state]
                        assert sum(learners[idx_particle]._state_counts_over_all_episodes) == len(learner._states)
                    else:
                        if use_fixed_average_reward:
                            info['average_reward'] = estimated_average_reward
                        self.learn_terminal_state_values(learner, t, state, next_state, reward, info, done_episode=next_state in self.env.getTerminalStates())
            else:
                # Step on the selected particle
                action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
                next_state, reward, done_episode, info = envs[idx_particle].step(action)

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
                if use_fixed_average_reward:
                    # NOTE: Setting this parameter to True ONLY has an effect when the stopping criterion is NOT any of the ones that include the MAX_TIME_STEPS condition
                    # (i.e. the stopping criterion is different from MAX_TIME_STEPS and MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES)
                    # because in those cases, the average reward is already fixed (because the average reward is only computed at the END of the FV simulation
                    # by the caller of this method, using a parametric estimate of P(T>t), which is based on ALL survival times (used in the estimation of E(T)).

                    ##### THIS SETTING OF THE AVERAGE REWARD TO A FIXED VALUE HAD BEEN A TEMPORARY PATCH TO TRY TO FIX THE UNLEARNING OF THE POLICY IN THE MOUNTAIN CAR PROBLEM, WHICH HAD WORKED! #########
                    # 2025/01/19: Use this as quick try of the REWARD SHAPING strategy of exit states to promote a policy that goes out of A.
                    # 2025/05/18: Also, it can be used to do the ABLATION study of estimating the average reward at a wrong value (e.g. always 0)
                    # DONE THE ABLATION STUDY TODAY AND SAW THAT THE POLICY IS NOT LEARNED AT ALL! (only for the state next to the EXIT state in a 6x8 gridworld with WIND=0.7).
                    ##### THIS SETTING OF THE AVERAGE REWARD TO A FIXED VALUE HAD BEEN A TEMPORARY PATCH TO TRY TO FIX THE UNLEARNING OF THE POLICY IN THE MOUNTAIN CAR PROBLEM, WHICH HAD WORKED! #########

                    # Always use the same (fixed) average reward value as correction of the value functions at every learning step
                    # Note that the check of whether estimated_average_reward is missing has been done at the beginning when parsing input parameters,
                    # in which case parameter use_fixed_average_reward is set to False.
                    info['average_reward'] = estimated_average_reward
                if is_learner_td_lambda:
                    # Use TD(lambda) on each particle separately, BUT using the COMMONLY estimated average reward (since we need all particles to do so)
                    info['average_reward'] = estimated_average_reward if use_fixed_average_reward else learner.getAverageReward()
                    learners[idx_particle].learn(t, state, action, next_state, reward, done, info)
                    # Update the trajectory and state counts stored in the base learner
                    # so that we can use them to analyze whether the absorption set A should be increased, based on the FV visits
                    # We pass `'learn_from_super_class': False` because we do NOT want to learn again, as learning just happened above
                    # with the call to learners[idx_particle].learn (notice that here we are calling the `learner` object, NOT the `learners[idx_particle]` object).
                    learner._states += [state]
                    assert sum(learners[idx_particle]._state_counts_over_all_episodes) == len(learner._states)
                else:
                    learner.learn(t, state, action, next_state, reward, done, info)
            if reward != 0.0:
                print(f"*** NON ZERO REWARD [1]!! (t={t}, P={idx_particle}, state={state} ({self.env.getStateFromIndex(state, simulation=False) if not self.env.isStateContinuous() else state}), action={action}, next_state={next_state} ({self.env.getStateFromIndex(next_state, simulation=False) if not self.env.isStateContinuous() else next_state}), reward={reward})")

            # Check if the particle is absorbed at the next state
            if not soft_killing and self.env.getIndexFromState(next_state) in absorption_set or \
               soft_killing and np.random.uniform() <= dict_proba_killing.get(next_state, 0.0):
                # The particle has been absorbed.
                # => Add the time to absorption to the set of times used to estimate the survival probability P(T>t) if it's the first absorption of the particle
                # => Reactivate the particle to the position of any of the other particles

                # Compute the absorption time as the time at which the particle is at *next_state*, and this is precisely the definition of t, as described above
                t_abs = t

                if is_learner_td_lambda:
                    # 14-Jul-2025: Reset the trajectory information and the eligibility traces information
                    # because these should NOT contain information from the particle BEFORE the latest absorption,
                    # as they do NOT correspond to transitions of the original Markov process whose value functions are of interest.
                    # HOWEVER, we should NOT reset the pieces of information that are normally reset at the end of an episode, that in the FV(Lambda) learning process
                    # are common to ALL particles, namely the state and action visit counts and the alpha learning rates, because these have to be determined by the OVERALL visits,
                    # o.w. we risk to have a too strong update of the value functions, making their estimates unstable.
                    learners[idx_particle]._reset_trajectory()
                    learners[idx_particle].reset_traces()

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

                    # NOTE: The iterative update of the average reward (Which involves an iterative update of the FV integral) has to be done BEFORE reactivating the particle,
                    # because the state of the particle AFTER reactivation is the state the particle takes STARTING at the current time; however, the FV integral at time t is
                    # the integral of P(T>t) Phi(t,x) UNTIL time t (excluded).
                    if  stopping_criterion_fv not in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES] \
                        and expected_absorption_time is not None:
                        # Only when the following two conditions are satisfied do the iterative update of the FV integral and of the average reward:
                        # - The stopping criterion is different from MAX_TIME_STEPS and MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES,
                        #   o.w. the average reward is computed at the END of the FV simulation (by the caller of this method), using a parametric estimate of P(T>t),
                        #   which is based on having observed ALL the survival times (used in the estimation of E(T)).
                        # - The expected re-absorption time has been given: because this value is needed for the computation of the average reward.
                        updated_average_reward = update_average_reward(learner, survival_times, expected_absorption_time, estimated_average_reward_at_start_of_fv_process=estimated_average_reward)
                        if plot:
                            # Compute the points to plot for the absorption set
                            self._update_plots(learner, t_learn, fig_V, fig_C, colors_V, len(colors_V.colors) if num_colors_in_colormap is None else num_colors_in_colormap,
                                               t, max_time_steps, points_to_add_in_counts_plot=list(absorption_set), pause=pause, method_name="_run_simulation_fv, ")
                                ## Notes:
                                ## - We might pass `None` to `fig_C`, the parameter after `fig_V` if we want a new figure to be generated (so that we see the evolution of the state counts)
                                ## - The parameter after `colors_V` defining the number of colors in the colormap is extracted from the length of attribute colors_V.colors
                                ## ONLY if num_colors_in_colormap is None (which is used as `lut=` parameter of the cm.get_cmap() function creating the colormap)
                                ## because of num_colors_in_colormap is specified (e.g. 10), colors_V does NOT have any `colors` attribute!!!!

                            dict_lines = update_plots(dict_axes, dict({'average_reward': [(t_surv, updated_average_reward),
                                                                                          (t_surv, learner.getAverageReward())]}), dict_lines, show_title=False)
                            # Make sure that the Y axis is re-scaled after adding the new point (because after setting the axis limits (as done right below with set_ylim()),
                            # the axis limit is NOT re-calculated after adding new points (see: https://matplotlib.org/stable/users/explain/axes/autoscale.html))
                            dict_axes['average_reward'].autoscale(axis='y')
                            if learner.getAverageReward() < 0:
                                # We assume the rewards are negative
                                # => Set the MAXIMUM at 0, so that we get a visual comparison with other figures plotting the same concept
                                dict_axes['average_reward'].set_ylim((None, 0))
                            else:
                                # We assume the rewards are positive
                                # => Set the minimum at 0, so that we get a visual comparison with other figures plotting the same concept
                                dict_axes['average_reward'].set_ylim((0, None))
                            plt.figure(dict_axes['average_reward'].get_figure().number)  # Need to select the figure so that it is re-drawn, otherwise, the redraw applies to the active figure (already redrawn) selected above by plt.figure()
                            plt.suptitle(f"[_run_simulation_fv, Learning step {t_learn+1}]\nSurvival time #{n_particles_absorbed_once} out of max possible {N} with survival time t_surv={t_surv}")
                            plt.pause(pause)
                            plt.draw()

                    # Show the progress of particles absorption
                    if int((n_particles_absorbed_once - 1) / N * 100) % 10 != 0 and int(n_particles_absorbed_once / N * 100) % 10 == 0:
                        print("t={} of {} of {}: {:.1f}% of particles absorbed at least once ({} of {})".format(t, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once / N * 100, n_particles_absorbed_once, N))
                        # Take the opportunity to plot the distribution of the particles
                        if plot:  #False
                            # Plot the distribution of the FV particles in the environment
                            _state_counts = np.zeros(self.env.getNumStates())
                            _dist_fv_particles = pd.Series([env.getState() for env in envs]).value_counts(sort=False)
                            _state_counts[_dist_fv_particles.index] = _dist_fv_particles
                            self.env.plot_values(_state_counts, ax=ax_dist_fv, cmap="Blues", vmin=0, vmax=N, add_colorbar=False)
                            ax_dist_fv.set_title(f"Learning step t_learn = {t_learn+1}"
                                                 f"\nSTART STATE CASE: '{start_state_selection_case}'"
                                                 f"\nDist. of FV particles (N={N}) at t = {t} of {max_time_steps}"
                                                 f" ({n_particles_absorbed_once / N * 100:.0f}% absorbed once)", fontsize=7)
                            plt.pause(0.01)
                            plt.draw()
                    if False:
                        print("Survival times observed so far: {}".format(n_particles_absorbed_once))
                        print(survival_times)

                # Reactivate the particle
                _absorbed_state = next_state
                next_state = reactivate_particle_internal(idx_particle)
                reward = envs[idx_particle].getReward(next_state)
                rewards_after_reactivation.append(reward)
                if reward != 0.0:
                    print(f"*** [reactivation] NON ZERO REWARD [2]!! (t={t}, P={idx_particle}, state={state} ({self.env.getStateFromIndex(state, simulation=False) if not self.env.isStateContinuous() else state}) -> "
                          f"abs_state={_absorbed_state} ({self.env.getStateFromIndex(_absorbed_state, simulation=False) if not self.env.isStateContinuous() else _absorbed_state}), action='REACTIVATION', "
                          f"next_state={next_state} ({self.env.getStateFromIndex(next_state, simulation=False) if not self.env.isStateContinuous() else next_state}), reward={reward})")
                if DEBUG_TRAJECTORIES:
                    print(f"--> Reactivated particle {idx_particle} from state {_absorbed_state} ({self.env.getIndexFromState(_absorbed_state)}) to state {next_state} ({self.env.getIndexFromState(next_state)}) (reward={reward} received)")
                if not soft_killing:
                    assert next_state not in absorption_set, f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

            # Update Phi based on the new state of the changed (and possibly also reactivated) particle
            if expected_absorption_time is not None:
                # Iterative update of Phi
                # TODO: (2024/07/21) Fix the following, based on the comment I wrote on 2024/05/22 written here
                # (2024/05/22) ACTUALLY, this also calls update_phi() as done below in the ELSE block... the only difference is that alpha is None in this case,
                # so I don't know why I separate these two cases. In fact, they are exactly the same if alpha0 = +np.Inf, which is often a good choice.
                learner.update_phi(t, state, next_state)
            else:
                # One-shot update of Phi
                learner.dict_phi = update_phi(envs[0], len(envs), t, learner.dict_phi, state, next_state, alpha=alpha0 / (t + alpha0))

            if DEBUG_TRAJECTORIES:
                print("[FV] Moved P={}, t={}: state={}, action={} -> next_state={}, reward={}" \
                      .format(idx_particle, t, state, action, next_state, reward),
                      end="\n")
                if N <= 10:
                    print(f"Current state of system:\n{[env.getState() for env in envs]}")

            idx_reactivate = None

            # CHECK CHANGE OF SYSTEM'S STATE
            # Check whether the system state has changed after updating the selected particle
            # Goal: Detect a possible "get stuck" situation that could arise when the absorption set A is not connected
            # (e.g. 2D labyrinth with corridor where the start state is part of the absorption set)
            if np.allclose(next_state, state):  # We use np.allclose() and NOT an equality because the state can be multidimensional and continuous-valued (e.g. Mountain Car)
                n_consecutive_steps_at_same_system_state += 1
            else:
                n_consecutive_steps_at_same_system_state = 0
            if n_consecutive_steps_at_same_system_state > 0 and n_consecutive_steps_at_same_system_state % 20 == 0:
                    ## Note: we check for `MOD 20` and NOT `>= 20` because we don't want to show this message all the time (i.e. for values 20, 21, 22, etc.)
                    ## but just when a new block of 20 consecutive time steps in this situation is observed again.
                warnings.warn(f"The FV system's state hasn't changed for {n_consecutive_steps_at_same_system_state} consecutive steps."
                              "\nThis may happen when the absorption set A is not connected and the start state distribution on the outside boundary of A is very uneven."
                              "\nExample of such situation is: 2D labyrinth with corridor where the start state is part of the absorption set A, and the policy is close to optimal."
                              f"\nThe repeated system state is:\n{[env.getState() for env in envs]}\n{[env.getStateFromIndex(env.getState(), simulation=False) for env in envs]}")

            # CHECK DONE
            # Note that this stopping criterion impacts ONLY the estimation of the FV-based AVERAGE REWARD, but it does NOT impact the estimation of the value functions
            # done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV (but instead we are using FV just as a oversampling mechanism),
            # we may want to continue until we reach the maximum number of steps, regardless of the first absorption event of each particle.
            if stopping_criterion_fv == StoppingCriterion.MAX_TIME_STEPS:
                # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached (which implies that max_time_steps must NOT be infinite).
                # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
                # because we guarantee that ALL time steps specified for the FV learner are used (and this number of time steps normally comes from the FAIR comparison setup).
                # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
                done = t >= max_time_steps
            elif stopping_criterion_fv == StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES:
                # Condition for stopping, either when:
                # a) a pre-specified proportion of N particles have been absorbed at least once
                # b) the maximum simulation time has been reached (max_time_steps)
                done = n_particles_absorbed_once >= min_prop_absorbed_particles * N or \
                       t >= max_time_steps  # max_time_steps is the LARGEST of the max times (i.e. max_time_steps >= max_time_steps_for_absorbed_particles_check)
            elif stopping_criterion_fv == StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN:
                # Condition for stopping, either when:
                # a) all N particles have been absorbed at least once
                # (because this completes the estimation of P(T>t) making its value = 0 for t > the time of the last particle absorption
                # b) the maximum simulation time for checking the proportion of absorbed particles has been reached and overpassed
                # (this makes sense only when max_time_steps_for_absorbed_particles_check is NOT infinite), AND a large enough number of particles (e.g. 90%)
                # has been absorbed at least once.
                # c) the maximum simulation time has been reached (max_time_steps)
                done = n_particles_absorbed_once == N or \
                       t >= max_time_steps_for_absorbed_particles_check and n_particles_absorbed_once >= min_prop_absorbed_particles*N or \
                       t >= max_time_steps  # max_time_steps is the LARGEST of the max times (i.e. max_time_steps >= max_time_steps_for_absorbed_particles_check)
            elif stopping_criterion_fv == StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES:
                # Condition for stopping, whatever happens later: the maximum allowed number of steps or the absorption of the given proportion of particles
                # The goal here is to leverage the largest amount of information that can be provided by the FV particle system within a flexible budget of max_time_steps
                done = t >= max_time_steps and n_particles_absorbed_once >= min_prop_absorbed_particles*N
            else:
                raise ValueError(f"Invalid value given for `stopping_criterion_fv`: {stopping_criterion_fv}")

        # DONE
        if show_messages(verbose, verbose_period, t):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps_for_absorbed_particles_check={} of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps_for_absorbed_particles_check*100, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, max(survival_times)))
        if n_particles_absorbed_once < N:
            # There are still particles that have not been absorbed at least once
            # => Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - n_particles_absorbed_once
            survival_times += list(np.repeat(t, n_particles_not_absorbed))
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE during the time allowed for the FV simulation"
                      f" ({max_time_steps_for_absorbed_particles_check} or {max_time_steps} has been reached):"
                      f" # NOT absorbed particles = {n_particles_not_absorbed} ({np.round(n_particles_not_absorbed / N * 100, 1)}%)")

            # Update the iterative computation of the FV integral and stationary probabilities
            if expected_absorption_time is not None:
                update_average_reward(learner, survival_times, expected_absorption_time, estimated_average_reward_at_start_of_fv_process=estimated_average_reward)
        if stopping_criterion_fv in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES]:
            # These stopping criterion will use a parametric estimation of the survival probability distribution P(T>t), therefore no merge will be done between P(T>t) and Phi(x,t)
            # => Complete the measurement of Phi(x,t) up to the last simulation time observed so that the last piece (which may be very long, especially in the cases when
            # the FV particles get stuck at the same configuration and this configuration involves many (or all) particles at the state with reward).
            for x in learner.dict_phi.keys():
                _phi_last = learner.dict_phi[x]['Phi'].iloc[-1]
                learner.dict_phi[x] = pd.concat([learner.dict_phi[x],
                                                 pd.DataFrame([[t, _phi_last]], index=[learner.dict_phi[x].shape[0] + 1], columns=['t', 'Phi'])])

        # Assertions about survival times and maximum time stored in the Phi dictionary
        assert len(survival_times) - 1 == N, f"The number of elements stored in the `survival_times` list ({len(survival_times)}) must be N+1 ({N + 1}) at the end of the FV simulation"
        if stopping_criterion_fv not in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES]:
            # The following assertion should be used ONLY when the condition on the absorption of the N particles at least once can stop the simulation,
            # independently of the number of simulation steps taken.
            # In fact, if this is not the case the simulation can go on even if all N particles have been absorbed at least once
            # making the times stored in Phi(t, x) be possibly LARGER than the maximum observed survival time, which is what is checked here.
            assert np.max([np.max(learner.dict_phi[x]['t']) for x in learner.dict_phi.keys()]) <= np.max(survival_times), \
                    f"Under an FV stopping criterion OTHER THAN by number of steps, the maximum time stored in Phi(t,x) must be at most the maximum observed survival time"

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        # Store the RAW average reward observed by the FV particles which can be used to evaluate the oversampling effect generated by the FV particle system
        learner.setAverageRewardRaw( np.mean(learner._rewards + rewards_after_reactivation) )

        # Update the absorption set by adding frequently visited states during the FV simulation with no reward, so that the agent can get closer to the states with rewards
        if update_absorption_set_with_fv_visits:
            # Remove any reward shaping reward so that we can correctly select new states with no reward to enlarge the absorption set A
            if reward_shaping:
                for s in self.env.getAllValidStates():
                    # TODO: (2025/01/20) Generalize the following reset of the rewards to zero for cases where non-zero rewards can also be observed in non-terminal states
                    # IMPORTANT: It is assumed that all states except terminal states have ZERO REWARD! This may not be the case in the future...
                    if not self.env.isTerminalState(s):
                        self.env.setReward(s, 0.0)

            # Compute the distribution of the states visited by the FV particle system (under VALID transitions)
            # so that we can add the most frequently visited states to the absorption set!
            _max_cum_freq_threshold = 0.50; _min_freq_threshold = 0.50
            print(f"ABSORPTION SET: Analyzing its update with new states, based on FV minimum visit frequency of {_min_freq_threshold*100}% or maximum cumulative visit frequency threshold of {_max_cum_freq_threshold*100}%...")
            _state_indices = learner._states
            # IMPORTANT: In the FV simulation, there is NO correspondence between the states collected in learner._states and the rewards collected in learner._rewards,
            # because particles are selected at random for update, therefore the nice correspondence _states[t] <-> _rewards[t], i.e. where _rewards[t] is the reward received
            # when visiting state _states[t] that exists in the single Markov chain simulation case, is no longer true in the FV simulation, because _states[t] corresponds to the
            # state of the particle picked NEXT, whose reward has nothing to do with _rewards[t] which was added to the list when visiting _states[t-1] which is the state
            # of ANOTHER particle.
            # This is why here we need to compute the reward associated to each state in order to retrieve the reward associated to each state visited by the FV particles.
            _state_indices_with_zero_reward = [s for s in _state_indices if self.env.getReward(self.env.getStateFromIndex(s, simulation=True)) == 0.0]
            _dist_state_counts = pd.Series(_state_indices_with_zero_reward).value_counts(normalize=True)     # Note: the distribution is naturally sorted by decreasing frequency
            _cum_dist_state_counts = np.cumsum(_dist_state_counts)
            _new_absorption_set = set(_cum_dist_state_counts.index[(_cum_dist_state_counts <= _max_cum_freq_threshold) | (_dist_state_counts >= _min_freq_threshold)])
            _current_length_of_absorption_set = len(absorption_set)
            absorption_set = absorption_set.union(_new_absorption_set)
            _new_length_of_absorption_set = len(absorption_set)
            less_frequently_visited_set = set(_dist_state_counts.index).difference(_new_absorption_set)
            if _new_length_of_absorption_set - _current_length_of_absorption_set > 0:
                print(f"--> The absorption set will be potentially updated (given certain conditions are satisfied) "
                      f"with {_new_length_of_absorption_set - _current_length_of_absorption_set} new states "
                      f"(growing {(_new_length_of_absorption_set / _current_length_of_absorption_set - 1):.1f}%, "
                      f"from {_current_length_of_absorption_set} to {_new_length_of_absorption_set} states)")
            else:
                print(f"--> The absorption set will NOT be updated as no new states were visited under the above conditions.")
        else:
            # When the absorption set is not updated, we set the set of less frequently visited states to all the states visited during the FV simulation
            # so that any of them can be selected as start states for the next FV simulation (for the next policy learning step), in case "everything else" fails.
            # Note that this may or may not be used in the code, depending on the current state of the evolving implementation.
            less_frequently_visited_set = set(learner._states)

        if DEBUG_ESTIMATORS and stopping_criterion_fv not in [StoppingCriterion.MAX_TIME_STEPS, StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES]:
            # Print and plot the estimators of P(T>t) and Phi(x,t) only when the stopping criterion is not among those for which the FV integral is computed at the end of the simulation
            # as in that case this print and plot is generated by the estimate_stationary_probabilities() function in estimators/fv.py.
            _pandas_options = set_pandas_options()
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(dict_phi[x]))
            reset_pandas_options(_pandas_options)

            # Make a plot
            plt.figure()
            plt.step(df_proba_surv['t'], df_proba_surv['P(T>t)'], color="blue", where='post')
            for x in sorted(learner.dict_phi.keys()):
                # Choose a particular state to plot if needed
                # if x == 263: #19:
                if self.env.getReward(x) != 0.0:
                    # Parametric survival function on the times measured by both P(T>t) and Phi (considering either of them is not enough because of their different change times)
                    alpha = 1 / np.mean(df_proba_surv['t'])
                    # K-S test of goodness of fit
                    import scipy.stats as stats
                    ks_test = stats.kstest(df_proba_surv['t']*alpha, "expon")
                    # Plot the parametric survival function estimate
                    _t = sorted(np.r_[df_proba_surv['t'], learner.dict_phi[x]['t']])
                    plt.plot(_t, np.exp(-alpha * _t), color="blue", linestyle="dashed", linewidth=2)
                    # Plot Phi(x,t)
                    plt.step(learner.dict_phi[x]['t'], learner.dict_phi[x]['Phi'], color="darkviolet", where='post')
                    plt.title(f"[_run_simulation_fv, Learning step {t_learn + 1}]"
                              f"\nP(T>t) (blue) and Phi(t,x) (violet) for state x = {x} ({self.env.getStateFromIndex(x, simulation=False) if not self.env.isStateContinuous() else x})"
                              f"\nKS test for goodness of fit with exponential(alpha={alpha:.3f}): p-value = {ks_test.pvalue:.3g}")
                    plt.pause(0.001)
                    plt.draw()

        if plot:
            self._update_plots_at_episode_end(0, 1, learner, t_learn, fig_V, fig_V2, None, colors_V, None, 0.0, pause=pause, method_name="_run_simulation_fv, ")
            self._final_plots(learner, t_learn, fig_V, fig_C, points_to_add_in_counts_plot=list(absorption_set), method_name="_run_simulation_fv, ")

        return t, learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner._state_counts, \
               learner.dict_phi, df_proba_surv, expected_absorption_time, max_survival_time, \
               absorption_set, less_frequently_visited_set

    @measure_exec_time
    # DM-2025/02/03: Deprecated method because the signature of the MOTHER method _run_simulation_fv() changed from `stop_if_prop_absorbed_particles_reached_regardless_of_time_steps` to `stopping_criterion_fv`.
    def _deprecated_run_simulation_fv_fraiman( self, t_learn, envs, absorption_set: set, start_set: set,
                                    max_time_steps=None,
                                    max_time_steps_for_absorbed_particles_check=+np.Inf, min_prop_absorbed_particles=0.90, stop_if_prop_absorbed_particles_reached_regardless_of_time_steps=False,
                                    dist_proba_for_start_state: dict=None,
                                    expected_absorption_time=None, expected_exit_time=None,
                                    estimated_average_reward=None,
                                    epsilon_random_action=0.0,
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

        # Set the seed of the environment stored in the policy which is the one responsible for defining the next action of the agent
        # Note that this environment normally coincides with the environment stored in this Simulator object, but it may not always be the case
        # (this already happened when I was using different COPIES of a policy to compare different value function learners! May-2024)
        if seed is not None:
            policy.env.seed(seed)

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        # Note that also the state visit COUNTS are reset, therefore we lose the information of visits observed during the initial exploration (typically run by a call to _run_single_continuing_task())
        learner.reset(reset_episode=True, reset_value_functions=False, reset_average_reward=estimated_average_reward is None)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            # Environment seed
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the activation set
            if dist_proba_for_start_state is not None and start_set != set(dist_proba_for_start_state.keys()):
                warnings.warn(f"[_run_simulation_fv_fraiman] The set of start states given in `start_set` ({start_set}) "
                              f"is NOT equal to the keys present in the dictionary containing the start state distribution to use `dist_proba_for_start_state`:\n{dist_proba_for_start_state})"
                              f"\nThe start states will be selected UNIFORMLY AT RANDOM (but this may not be what is wished).")
                dist_proba_for_start_state = None
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

        # Phi(t, z): Empirical probability of ALL active states z, at each time t when a variation in Phi(t, z) is observed.
        # Note that we need to estimate Phi(t, z) for all active states because we use it as the probability distribution to reactivate ALL absorbed particles at one time.
        dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getActiveSet())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = deque([0])   # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        has_particle_been_absorbed_once = [False]*N   # List that keeps track of whether each particle has been absorbed once
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        n_particles_absorbed_once = 0
        # Times at which each particle visited an activation set state, so that we can measure the survival time of absorptions beyond the first one
        times_last_visit_activation_set = [0]*N
        # Times at which each particle was reactivated, which is used to decide whether we need to update the respective time of the last visit to the activation set (see logic explanation below)
        times_last_absorption = [0]*N

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        done = False
        t = 0   # Learning step counter: t represents the time at which a particle transitions to the NEXT state. See more details at the @note at the beginning of the file.
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
                    action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
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
                        n_particles_absorbed_once += 1
                        if False:
                            print("Survival times observed so far: {}".format(n_particles_absorbed_once))
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

            print("t={} of {} of {}: % particles absorbed at least once = {:.1f}% ({} of {})".format(t, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, n_particles_absorbed_once, N))

            # Update Phi after all particles have been updated
            dict_phi = update_phi_on_all_states(envs, t, dict_phi, alpha=alpha0 / (t + alpha0))

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER:
            # - when all N particles have been absorbed at least once (because this completes the estimation of P(T>t) making its value = 0 for t > the time of the last particle absorption
            # OR
            # - when the maximum simulation time for checking the proportion of absorbed particles has been reached and overpassed
            # (makes sense only when max_time_steps_for_absorbed_particles_check is NOT infinite)
            # AND having a large enough number of particles (e.g. 90%) absorbed at least once.
            # Note that this logic impacts ONLY the estimation of the FV-based AVERAGE REWARD, but it does NOT impact the estimation of the value functions
            # done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles (or a percentage of them) have been absorbed at least once
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles (or a percentage of them) have been absorbed at least once
            if stop_if_prop_absorbed_particles_reached_regardless_of_time_steps:
                done = n_particles_absorbed_once >= min_prop_absorbed_particles * N or \
                       t >= max_time_steps  # max_time_steps is the LARGEST of the max times (i.e. max_time_steps >= max_time_steps_for_absorbed_particles_check)
            else:
                done = n_particles_absorbed_once == N or \
                       t >= max_time_steps_for_absorbed_particles_check and n_particles_absorbed_once >= min_prop_absorbed_particles*N or \
                       t >= max_time_steps
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if n_particles_absorbed_once < N:
            # Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - n_particles_absorbed_once
            for e in np.repeat(t, n_particles_not_absorbed):
                survival_times.append(e)
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE during the time allowed for the FV simulation" +
                      (stop_if_prop_absorbed_particles_reached_regardless_of_time_steps and f" ({max_time_steps} has been reached):" or f" ({max_time_steps_for_absorbed_particles_check} or {max_time_steps} has been reached):") +
                      f" # NOT absorbed particles = {n_particles_not_absorbed} ({np.round(n_particles_not_absorbed / N * 100, 1)}%)")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(dict_phi[x]['t']) for x in dict_phi.keys()]) <= np.max(survival_times), "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps_for_absorbed_particles_check={} of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps_for_absorbed_particles_check*100, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, survival_times[-1]))

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

        return n_events_fv, learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner._state_counts, dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    @measure_exec_time
    # DM-2025/02/03: Deprecated method because the signature of the MOTHER method _run_simulation_fv() changed from `stop_if_prop_absorbed_particles_reached_regardless_of_time_steps` to `stopping_criterion_fv`.
    def _deprecated_run_simulation_fv_fraiman_modified(self, t_learn, envs, absorption_set: set, start_set: set,
                                            max_time_steps=None,
                                            max_time_steps_for_absorbed_particles_check=+np.Inf, min_prop_absorbed_particles=0.90, stop_if_prop_absorbed_particles_reached_regardless_of_time_steps=False,
                                            dist_proba_for_start_state: dict=None,
                                            expected_absorption_time=None, expected_exit_time=None,
                                            estimated_average_reward=None,
                                            epsilon_random_action=0.0,
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

        # Set the seed of the environment stored in the policy which is the one responsible for defining the next action of the agent
        # Note that this environment normally coincides with the environment stored in this Simulator object, but it may not always be the case
        # (this already happened when I was using different COPIES of a policy to compare different value function learners! May-2024)
        if seed is not None:
            policy.env.seed(seed)

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False, reset_average_reward=estimated_average_reward is None)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            # Environment seed
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the activation set
            if dist_proba_for_start_state is not None and start_set != set(dist_proba_for_start_state.keys()):
                warnings.warn(f"[_run_simulation_fv_fraiman_modified] The set of start states given in `start_set` ({start_set}) "
                              f"is NOT equal to the keys present in the dictionary containing the start state distribution to use `dist_proba_for_start_state`:\n{dist_proba_for_start_state})"
                              f"\nThe start states will be selected UNIFORMLY AT RANDOM (but this may not be what is wished).")
                dist_proba_for_start_state = None
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

        # Phi(t, z): Empirical probability of ALL active states z, at each time t when a variation in Phi(t, z) is observed.
        # Note that we need to estimate Phi(t, z) for all active states because we use it as the probability distribution to reactivate ALL absorbed particles at one time.
        dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getActiveSet())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = deque([0])   # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        has_particle_been_absorbed_once = [False]*N     # List that keeps track of whether each particle has been absorbed once
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        n_particles_absorbed_once = 0
        has_particle_been_absorbed = [False]*N          # This is used to determine what particles are used to compute the reactivation distribution, namely the non-absorbed ones

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        done = False
        t = 0   # Learning step counter: t represents the time at which a particle transitions to the NEXT state. See more details at the @note at the beginning of the file.
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
                    action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
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
                        n_particles_absorbed_once += 1
                        if False:
                            print("Survival times observed so far: {}".format(n_particles_absorbed_once))
                            print(survival_times)

            # Reactivate absorbed particles
            if n_particles_absorbed_once == N:
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

            print("t={} of {} of {}: % particles absorbed at least once = {:.1f}% ({} of {})".format(t, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, n_particles_absorbed_once, N))

            # Update Phi after all particles have been updated
            dict_phi = update_phi_on_all_states(envs, t, dict_phi, alpha=alpha0 / (t + alpha0))

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER:
            # - when all N particles have been absorbed at least once (because this completes the estimation of P(T>t) making its value = 0 for t > the time of the last particle absorption
            # OR
            # - when the maximum simulation time for checking the proportion of absorbed particles has been reached and overpassed
            # (makes sense only when max_time_steps_for_absorbed_particles_check is NOT infinite)
            # AND having a large enough number of particles (e.g. 90%) absorbed at least once.
            # Note that this logic impacts ONLY the estimation of the FV-based AVERAGE REWARD, but it does NOT impact the estimation of the value functions
            # done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles (or a percentage of them) have been absorbed at least once
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles (or a percentage of them) have been absorbed at least once
            if stop_if_prop_absorbed_particles_reached_regardless_of_time_steps:
                done = n_particles_absorbed_once >= min_prop_absorbed_particles * N or \
                       t >= max_time_steps  # max_time_steps is the LARGEST of the max times (i.e. max_time_steps >= max_time_steps_for_absorbed_particles_check)
            else:
                done = n_particles_absorbed_once == N or \
                       t >= max_time_steps_for_absorbed_particles_check and n_particles_absorbed_once >= min_prop_absorbed_particles*N or \
                       t >= max_time_steps
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if n_particles_absorbed_once < N:
            # Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - n_particles_absorbed_once
            for e in np.repeat(t, n_particles_not_absorbed):
                survival_times.append(e)
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE during the time allowed for the FV simulation" +
                      (stop_if_prop_absorbed_particles_reached_regardless_of_time_steps and f" ({max_time_steps} has been reached):" or f" ({max_time_steps_for_absorbed_particles_check} or {max_time_steps} has been reached):") +
                      f" # NOT absorbed particles = {n_particles_not_absorbed} ({np.round(n_particles_not_absorbed / N * 100, 1)}%)")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(dict_phi[x]['t']) for x in dict_phi.keys()]) <= np.max(survival_times), "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps_for_absorbed_particles_check={} of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps_for_absorbed_particles_check*100, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, survival_times[-1]))

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

        return n_events_fv, learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner._state_counts, dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    @measure_exec_time
    # DEPRECATED-2024/08/07: Following the elimination of parameter `estimate_on_fixed_sample_size` from the LeaFV constructor,
    # this method was deprecated because it still makes reference to parameter `estimate_on_fixed_sample_size`
    # whose value does no longer coincide with the new attribute stored in the LeaFV learner, `is_criterion_discounted`,
    # as the parameter can be set to True or False while still learning under the DISCOUNTED criterion.
    # Another reason for deprecation is that doing FV in the discounted context as implemented here is highly computationally intensive
    # and it therefore doesn't make sense to be used.
    def _deprecated_run_simulation_fv_discounted(  self, t_learn, envs, absorption_set: set, start_set: set,
                                        max_time_steps=None,
                                        max_time_steps_for_absorbed_particles_check=+np.Inf, min_prop_absorbed_particles=0.90,
                                        number_free_particles=1,
                                        dist_proba_for_start_state: dict=None,
                                        expected_absorption_time=None, expected_exit_time=None,
                                        estimated_average_reward=None,
                                        epsilon_random_action=0.0,
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

        # Set the seed of the environment stored in the policy which is the one responsible for defining the next action of the agent
        # Note that this environment normally coincides with the environment stored in this Simulator object, but it may not always be the case
        # (this already happened when I was using different COPIES of a policy to compare different value function learners! May-2024)
        if seed is not None:
            policy.env.seed(seed)

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False, reset_average_reward=estimated_average_reward is None)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        df_particles_for_start_state = None
        particles_first_action = [-1]*N
        if learner.estimate_on_fixed_sample_size:
            # Create a data frame to store the pieces of information that are necessary to perform the FV estimation of value functions using a fixed sample size of particles
            # defined by the number of particles that start at each possible state and state-actions where the value functions should be estimated.
            # These states and state-actions are those corresponding to the states belonging to the active set, and this is why we index each row of the data frame
            # with the different states in the active set, as defined in the FV learner.
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
            if dist_proba_for_start_state is not None and start_set != set(dist_proba_for_start_state.keys()):
                warnings.warn(f"[_run_simulation_fv_discounted] The set of start states given in `start_set` ({start_set}) "
                              f"is NOT equal to the keys present in the dictionary containing the start state distribution to use `dist_proba_for_start_state`:\n{dist_proba_for_start_state})"
                              f"\nThe start states will be selected UNIFORMLY AT RANDOM (but this may not be what is wished).")
                dist_proba_for_start_state = None
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
        n_particles_absorbed_once = 0
        has_particle_been_selected_once = [False]*N     # List that keeps track of whether each particle has been selected for movement at least once,
                                                        # so that we can choose the already selected start action when learning by groups

        if True or DEBUG_ESTIMATORS:
            if self.env.isStateContinuous():
                V = np.repeat(0.0, self.env.getNumStates())
                Q = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
            else:
                V = learner.getV().getValues()
                Q = learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(V))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(Q))

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
        info = dict()  # We need this variable to be defined when calling learner.learn() for the first time if the first particle picked for moving has started at a terminal state
        while not done:
            t += 1

            # Clock to use in the FV estimation of the value functions
            # In the fixed-sample-size estimation mode, this clock is the internal clock of the group to which the particle that is moved belongs to.
            t_clock = t

            if show_messages(verbose, verbose_period, t):
                print("@{}".format(get_current_datetime_as_string()))
                print(f"FV time step t={t} of max_time_steps_for_absorbed_particles_check={max_time_steps_for_absorbed_particles_check} of max_time_steps={max_time_steps} running...")
                print(f"# FV particles absorbed at least once: {n_particles_absorbed_once} of max N={N}")
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
            V_start_prev = self._get_state_value(learner, start_state)
            if learner.estimate_on_fixed_sample_size:
                # Update the particle's clock (used in the estimation of the survival probability P(T>t) for each start state)
                particle_times[idx_particle] += 1

                # Update the group's clock (used in the estimation of the conditional occupation probability Phi(t,x) for each start state
                df_particles_for_start_state.loc[start_state, 't'] += 1
                t_clock = df_particles_for_start_state.loc[start_state, 't']

            # Get the current state of the selected particle because that's the particle whose state is going to (possibly) change
            state = envs[idx_particle].getState()
            V_state_prev = self._get_state_value(learner, state)  # Store the current value of V(s) as PREVIOUS value for informative plotting purposes (title)
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
                    action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
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
                # TO-DO: (2024/01/29) Revise the correct use of the `done` variable here, instead of `done_episode`, because actually when we are done by `done`, this line will NEVER be executed because we will NOT enter again the `while done` loop...
                assert state in learner.getActiveSet()
                if True or df_particles_for_start_state is None or state not in df_particles_for_start_state.index:
                    info['update_counts'] = True #True #False
                    learner.learn(t_clock, state, action, next_state, reward, done, info, envs=envs, idx_particle=idx_particle, update_phi=True)
                    info.pop('update_counts')

            # Step on all the "normal" particles that have been created for the exploration of the system following the dynamics of the *underlying* Markov chain
            for idx_env, env in enumerate(envs_normal):
                #print(f"\n[debug] Moving normal particle #{idx_env}...")
                state_normal = env.getState()
                action_normal = self._choose_action(policy, state_normal, epsilon_random_action=epsilon_random_action)
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
                    n_particles_absorbed_once += 1

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
                        print("First-time survival times observed so far: {}".format(n_particles_absorbed_once))
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
                        plt.title(f"[_run_simulation_fv_discounted, Learning step {t_learn+1}] V(s) evolution (blueish: initial, reddish: final)"
                                  f"\nSystem's time t = {t}, Abs. no. = {n_absorptions} (# first abs. = {n_particles_absorbed_once} of max N = {N}), start_state = {start_state}"
                                  f"\nfrom state = {state} --> absorption state = {state_absorption} (V(s)={np.round(self._get_state_value(learner, state_absorption), 3)}) --> reactivation state = {next_state}"
                                  f"\nV(start={start_state}) = {np.round(V_start_prev, 3)}) --> V(start={start_state}) = {np.round(self._get_state_value(learner, start_state), 3)} (delta(V) = {np.round(self._get_state_value(learner, start_state) - V_start_prev, 3)}, {np.round((self._get_state_value(learner, start_state) - V_start_prev) / max(1, V_start_prev) * 100, 1)}%)"
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
                plt.title(f"[_run_simulation_fv_discounted, Learning step {t_learn+1}] V(s) evolution (blueish: initial, reddish: final)"
                          f"\nSystem's time t = {t} of max {max_time_steps_for_absorbed_particles_check} of MAX {max_time_steps} (# first abs. = {n_particles_absorbed_once} of max N = {N}), start_state = {start_state}"
                          f"\nfrom state = {state} --> next state = {next_state} (V(s)={np.round(self._get_state_value(learner, next_state), 3)})"
                          f"\nV(state={state}) = {np.round(V_state_prev, 3)} --> V(state={state}) = {np.round(self._get_state_value(learner, state), 3)} (delta(V) = {np.round(self._get_state_value(learner, state) - V_state_prev, 3)}, {np.round((self._get_state_value(learner, state) - V_state_prev) / max(1, V_state_prev) * 100, 1)}%)"
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
            # If we want to interrupt the simulation by EITHER:
            # - when all N particles have been absorbed at least once (because this completes the estimation of P(T>t) making its value = 0 for t > the time of the last particle absorption
            # OR
            # - when the maximum simulation time for checking the proportion of absorbed particles has been reached and overpassed
            # (makes sense only when max_time_steps_for_absorbed_particles_check is NOT infinite)
            # AND having a large enough number of particles (e.g. 90%) absorbed at least once.
            # Note that this logic impacts ONLY the estimation of the FV-based AVERAGE REWARD, but it does NOT impact the estimation of the value functions
            # done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles (or a percentage of them) have been absorbed at least once
            done = n_particles_absorbed_once == N or \
                   t >= max_time_steps_for_absorbed_particles_check and n_particles_absorbed_once >= min_prop_absorbed_particles*(N - N_never_moved) or \
                   t >= max_time_steps
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if n_particles_absorbed_once < N - N_never_moved:
            # Add the last observed time step to the list of first-time survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - N_never_moved - n_particles_absorbed_once
            first_survival_times += list(np.repeat(t, n_particles_not_absorbed))
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N - N_never_moved} particles were absorbed at least ONCE during the smaller maximum number of time steps allowed for all particles"
                      f" ({max_time_steps_for_absorbed_particles_check} or {max_time_steps} has been reached):"
                      f" # NOT absorbed particles = {n_particles_not_absorbed} ({np.round(n_particles_not_absorbed / N * 100, 1)}%)")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(learner.dict_phi[x]['t']) for x in learner.dict_phi.keys()]) <= np.max(first_survival_times), \
                "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps_for_absorbed_particles_check={} of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps*100, max_time_steps_for_absorbed_particles_check, max_time_steps, n_particles_absorbed_once/N*100, first_survival_times[-1]))

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
                plt.step(learner.dict_phi[x]['t'], learner.dict_phi[x]['Phi'], color="darkviolet", where='post')
                plt.title(f"[_run_simulation_fv_discounted, Learning step {t_learn+1}]\nP(T>t) (blue) and Phi(t,x) (violet) for state x = {x}")

        print(f"Distribution of start actions:\n{pd.Series([learner.getStartAction(idx_particle) for idx_particle in range(N)]).value_counts()}")

        return n_steps_on_all_environments, learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner.getStateCounts(), learner.dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    def _run_single(self, nepisodes, t_learn=-1, max_time_steps=+np.Inf, max_time_steps_per_episode=+np.Inf, start_state_first_episode=None, reset_value_functions=True,
                    seed=None, compute_rmse=False, weights_rmse=None,
                    state_observe=None,
                    epsilon_random_action=0.0,
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

        t_learn: (opt) int
            The learning step number (starting at 0) for which the simulation is run when learning is used in the context of policy learning.
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.
            default: -1 (which means that the simulation is not part of a policy learning process by default)

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

        epsilon_random_action: (opt) float in [0, 1]
            Probability to take a random action instead of choosing an action dictated by the policy.
            This is useful to guarantee ergodicity when policies are deterministic or have some actions with zero probability.
            default: 0.0

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
            Name of the colormap to use in the generation of the animated plots showing the evolution of the value function estimates.
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
            fig_V, fig_V2, fig_C, fig_RMSE_state, colors_V, self.fig_policy = self._setup_plots(t_learn=t_learn, colormap=colormap, lut=nepisodes, state_observe=state_observe)
        #--- Parse input parameters

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()

        # Set seeds of:
        # - torch --> responsible for selecting the action when the policy is modeled via a neural network.
        # - the policy's environment --> just in case this environment does NOT have the same memory address as self.env
        #   (e.g. when policies are deepcopied from the original policy in order to compare different learning methods)
        # - numpy --> responsible for deciding whether a (completely) random action is chosen (when epsilon_random_action > 0).
        #   numpy is used to both draw a random number to decide whether to choose a random action (without following the policy)
        #   and then to actually choose that random action, if this ends up being the case.
        # - the object's environment --> responsible of deciding on the next state given the action.
        if seed is not None:
            torch.manual_seed(seed)
            policy.env.seed(seed)
            np.random.seed(seed)
            self.env.setSeed(seed)

        # Reset the environment (this should be done BEFORE resetting the learner because the learner will most likely store the reward at the initial state by calling
        # self.env.getReward() on the self.env.getState(), and if the state stored in the environment (from e.g. a previous execution/replication of the learning process),
        # is NOT the state at which the environment will start after reset below
        # Optional start state JUST for the very first episode
        # (In case we want to start at a specific state and then perform the subsequent resets as per the environment initial state distribution)
        self.env.reset()
        if start_state_first_episode is not None:
            self.env.setState(start_state_first_episode)
        if show_messages(verbose, verbose_period, 0):
            print("@{}".format(get_current_datetime_as_string()))
            print(f"[t_learn={t_learn}] Agent starts at state {self.env.getState()} with reward {self.env.getReward(self.env.getState())}")
        if self.debug:
            print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

        # Reset the learner (i.e. prepare it for a fresh new learning experience with all learning memory erased and learning rates reset to their initial values)
        # Note that a special treatment may be granted to the reset of the value functions because we may want NOT to reset them,
        # for instance when we are learning a policy and we use this simulator to learn the value functions...
        # In that case, we don't want to start off at 0.0 again but to start off at the estimated values obtained
        # under the previous policy, which normally is very close to the new policy after one step of policy learning.
        # (In some situations --e.g. labyrinth where the policy is learned using Actor-Critic or policy gradient learning--
        # I observed (Nov-2023) that non-optimal policies are learned that minimize the loss function if we reset the
        # value functions to 0 at every policy learning step, while the problem does NOT happen when the value functions are NOT reset.)
        learner.reset(reset_episode=True, reset_value_functions=reset_value_functions)

        # Store initial values used in the analysis of all the episodes run
        V_state_observe, RMSE, MAPE, ntimes_rmse_inside_ci95 = self._initialize_run_with_learner_status(nepisodes, learner, compute_rmse, weights, state_observe)

        # Initial state value function
        V = learner.getV().getValues()
        Q = learner.getQ().getValues()
        A = learner.getA().getValues()
        if verbose:
            print("Value functions at start of experiment:")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))
            print("A = {}".format(A.reshape(self.env.getNumStates(), self.env.getNumActions())))

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
        t = 0   # Learning step counter: t represents the time at which the Markov chain transitions to the NEXT state. See more details at the @note at the beginning of the file.
        episode = -1
        done = False
        while not done:
            episode += 1

            # Reset variables at the start of a new episode
            done_episode = False    # (2025/07/10) Note that here we set this variable to False and NOT to the result of checking whether the current state of the environment is a terminal state
                                    # (with `self.env.getState() in self.env.getTerminalStates()`) as done in other places, such as _run_single_continuing_task(),
                                    # because the environment is reset AFTER this (when episode > 0). And we cannot move this assignment of done_episode AFTER the reset of env
                                    # in the following `if episode > 0` block because the value of done_episode is passed to the self.learn_terminal_state_values() method.
            t_episode = -1          # Time step within the current episode
                                    # Note that we initialize it at -1 because the time within an episode indexes the time at which the ACTION is taken,
                                    # (and this will happen AFTER increasing t_episode by 1, so that the first time an action is taken will be indexed by t_episode = 0)
                                    # as opposed to the time at which the next state is observed (which is the convention used for variable `t` defined above).
                                    # This is done like that so that we have the state-action-reward sequence S(0), A(0), R(1), S(1), A(1), R(2), ...
                                    # So, t_episode indexes the state BEFORE transitioning.
            if episode > 0:
                # Reset the environment
                # Note that we do NOT reset the environment at the very first episode (episode = 0) because the environment has already been reset BEFORE starting the loop.
                self.env.reset()

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
                    # Note that when the end of the simulation has been reached, we ALSO set done_episode = True because, on one side this is conceptually true
                    # (i.e. an episode ends either when a terminal state is reached or when the maximum simulation time is reached), but ALSO because this allows
                    # to update the trajectory information stored in the learner (Learner._states, Learner._actions, Learner._rewards).
                    if max_time_steps is not None and t >= max_time_steps:
                        max_time_steps_reached = True
                        done_episode = True
                        if self.debug:
                            print("[run_single, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, episode+1))

                    # For the CONTINUING learning task context, we need to update the value of the state on which the previous episode ended
                    # because its value is not necessarily 0! (as long as this is NOT the first episode, which is the case at this point)
                    # (the value of a terminal state is 0 only in EPISODIC learning tasks, in which case it is 0 by definition of terminal states).
                    # In fact, in the continuing learning task, the environment state goes to a start state when the episode "terminates" and the Markov process continues.
                    reward = self.env.getReward(self.env.getState())
                    self.learn_terminal_state_values(learner, t_episode, terminal_state_previous_episode, self.env.getState(), reward, info, done_episode=done_episode)
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
                # IMPORTANT:
                # 1) We must reset the learner AFTER we reset the environment because the trajectory reset in the learner adds the first element to the learner._rewards
                # attribute which stores the reward at the initial state where the agent is located which should be the location of the agent after environment reset
                # (as opposed to the state where the environment was left by a potential previous replication run using the same learner)
                # 2) We only reset the learner for subsequent episodes (episode > 0) because the reset for episode 0 had already been done before entering the episode loop)
                # This is important because it actually affects the results of learning processes when the learning rate alpha is adjusted by the episode count,
                # as resetting the learner here again (for the first episode = 0) incorrectly increases the episode count which then affects the alpha adjustment!
                # (e.g. makes alpha be adjusted to alpha/3 instead of alpha/2 after the first episode was completed).
                learner.reset(reset_episode=False, reset_value_functions=False)

            while not done_episode:
                t += 1
                t_episode += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
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
                if max_time_steps is not None and t >= max_time_steps:
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
                    print("{:.4f}, Q=({},{})={:.4f}, avg. reward={:.4f}".format(learner.V.getValue(state), state, action, learner.Q.getValue(state, action), learner.getAverageReward()))
                if self.debug and done_episode:
                    print("--> [DEBUG] Done episode [{} iterations of {} per episode of {} total] at state {} with reward {}" \
                          .format(t_episode+1, max_time_steps_per_episode, max_time_steps, self.env.getState(), reward))

                # Observation state
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V_state_observe += [self._get_state_value(learner, state_observe)]
            #------- EPISODE FINISHED --------#

            # Store the value of the terminal state (used in the next episode when the average reward criterion is used for learning)
            terminal_state_previous_episode = next_state

            if show_messages(verbose, verbose_period, episode):
                print(", agent ENDS at state: {} at discrete time t = {})".format(self.env.getState(), t_episode+1), end=" ")
                print("")

            if not self.env.isStateContinuous():
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
                        ref_V_true, ref_V = self._compute_function_reference_values()
                        RMSE[min(episode+1, nepisodes)] = rmse(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)
                        MAPE[min(episode+1, nepisodes)] = mape(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)

                if plot and show_messages(True, verbose_period, episode):
                    self._update_plots_at_episode_end(episode, nepisodes, learner, t_learn, fig_V, fig_V2, fig_RMSE_state, colors_V, state_observe, ntimes_rmse_inside_ci95,
                                                      weights=weights, pause=pause, method_name="_run_single, ")

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

        if DEBUG_ESTIMATORS:
            if self.env.isStateContinuous():
                # We initialize DUMMY state value functions, so that we can refer to them as if the state space were discrete, so that we don't need to add many IF conditions below...
                # (2025/02/08) HOWEVER, maybe we should change this and call the self._get_state_value() and self._get_action_value() methods.
                V = np.repeat(0.0, self.env.getNumStates())
                Q = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
                A = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
            else:
                V = learner.getV().getValues()
                Q = learner.getQ().getValues()
                A = learner.getA().getValues()
            print(f"Estimated value functions at END of experiment (last episode run = {episode+1} of {nepisodes}):")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))
            print("A = {}".format(A.reshape(self.env.getNumStates(), self.env.getNumActions())))

        # Comment this out to NOT show the plot right away in case the calling function adds a new plot to the graph generated here
        if plot:
            self._final_plots(learner, t_learn, fig_V, fig_C, method_name="_run_single, ")

        if verbose:
            print("Percentage of episodes reaching max step = {:.1f}%".format(nepisodes_max_steps_reached / nepisodes*100))
            print("Last episode run = {} of {}".format(episode+1, nepisodes))

        return  learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner.getStateCounts(), RMSE, MAPE, \
                {   'nsteps': t,
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

    def _run_single_continuing_task(self, t_learn=-1, nepisodes=1, max_time_steps=1000, max_time_steps_per_episode=+np.Inf, start_state_first_episode=None,
                                    estimated_average_reward=None, use_fixed_average_reward=False, reset_value_functions=True,
                                    seed=None, compute_rmse=False, weights_rmse=None,
                                    state_observe=None, set_cycle=None, dict_proba_cycle=None,
                                    epsilon_random_action=0.0,
                                    reward_shaping=False, reward_for_exit_states: float=None,
                                    verbose=False, verbose_period=1, verbose_convergence=False,
                                    plot=False, colormap="seismic", pause=0.1):
        """
        Same as _run_single() but for CONTINUING learning tasks, i.e. where there are no episodes (although it works on fictitious episodes).

        At this point (Jan-2024), the method needs a LOT of refactoring, as this method was constructed keeping in mind that I didn't want to do too many changes
        to the _run_single() method from which it was created.
        For instance, we need to get rid of the WHILE loop on the episodes because the episodes never increase. So, at this point parameter nepisodes
        is expected to always be 1, as specified in its default value.

        t_learn: (opt) int
            The learning step number (starting at 0) for which the simulation is run when learning is used in the context of policy learning.
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.
            default: -1 (which means that the simulation is not part of a policy learning process by default)

        estimated_average_reward: (opt) None
            When given, it corresponds to the INITIAL estimation of the average reward used by the learner during the learning process.
            I.e. the average reward stored in the learner is NOT reset and then it is updated as new observations are collected during the current simulation.
            default: None

        use_fixed_average_reward: (opt) bool
            Whether to use the average reward given in `estimated_average_reward` (when not None or missing) as a fixed correction of the value functions
            estimated by the learner during this simulation.
            If True is given but the value of `estimated_average_reward` is missing, an error is raised.
            default: False

        set_cycle: (opt) set
            Set of states whose entrance from the complement set defines a cycle.
            Note that the set should include ALL the states, NOT only the boundary states through which the system can enter the set.
            The reason is that the set is used to determine which states are tracked for their visit frequency for the computation
            of their stationary probability using renewal theory.
            default: None

        dict_proba_cycle: (opt) dict
            Dictionary with the probability of starting a new cycle at each environment state.
            This is primarily used in the context of FV with SOFT killing, as this dictionary is responsible for defining the cycles to estimate E(T_A).
            It assumes that the environment state space is finite, as the state indexes the dictionary.
            For more details on how these probabilities are used to check whether a new cycle event occurs, see the
            _check_cycle_occurrence_and_update_cycle_variables method.
            default: None

        epsilon_random_action: (opt) float in [0, 1]
            Probability to take a random action instead of choosing an action dictated by the policy.
            This is useful to guarantee ergodicity when policies are deterministic or have some actions with zero probability.
            default: 0.0

        reward_shaping: (bool) opt
            Whether to do reward shaping of the EXIT states from the cycle set.
            This is typically used as a reward shaping strategy to nudge the policy towards visiting the boundary states of the absorption set A
            in FV estimation procedures, since those states are less frequently visited under the current policy, so we want to promote their visit
            to continue exploring outside the frequently visited states in A.
            default: False

        reward_for_exit_states: float
            Reward to assign to every exit state from the cycle set.
            Use np.nan to indicate that no reward shaping should be done even if reward_shaping=True.
            The use case for this is the following: show plots of the reward landscape for the current policy learning step and state that NO reward shaping
            is done (even if reward_shaping=True), which may be the case for instance of the absorption set A has not changed (i.e. grown)
            w.r.t. the previous absorption set, which indicates that we should stop doing reward shaping because A may have become large enough.
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
            - when one of parameters set_cycle or dict_proba_cycle is not None, two pieces of information that can be used to compute
            the stationary probability of states using its characterization based on renewal theory:
                - the number of cycles observed.
                - the time at which the process completed the last cycle.
                - the expected cycle time, i.e. the average cycle time where a cycle is defined by entering the cycle set after its latest exit.
                - an array with the visit count of all states.
            - A dictionary containing the distribution of the state at the start of the cycle.
            - A dictionary containing the distribution of the EXIT state from the cycle set. This is not empty only when set_cycle is not None and
            the excursion run here actually exited the cycle set at least once.
        """
        #--- Auxiliary functions
        def compute_proba_distribution_from_counts(dict_state_counts: dict, set_cycle: set=None):
            """
            Computes the state distribution based on their counts

            If set_cycle is not None, only states OUTSIDE the cycle set are considered in the distribution.
            This is needed when computing the probability distribution of the EXIT state from the cycle set, which is a set OUTSIDE the cycle set,
            whose states, when observed from outside the cycle set, are considered ENTRY states.
            """
            probas_state = dict()
            if len(dict_state_counts) > 0:
                if set_cycle is None:
                    # Add ALL states to the probability distribution calculation
                    probas_state = dict_state_counts.copy()
                else:
                    # Add ONLY the states OUTSIDE of the cycle set to the probability distribution calculation
                    for idx_state in dict_state_counts.keys():
                        if idx_state not in set_cycle:
                            probas_state[idx_state] = dict_state_counts[idx_state]

                # Compute the probabilities and check they sum up to 1
                _total_number_of_cases = sum(probas_state.values())
                for idx_state in probas_state.keys():
                    probas_state[idx_state] /= _total_number_of_cases
                assert np.isclose(sum(probas_state.values()), 1.0)

            return probas_state
        #--- Auxiliary functions

        #--- Parse input parameters
        # ---- UPDATE FOR CONTINUING TASK
        if max_time_steps is None or max_time_steps <= 0 or max_time_steps == np.Inf:
            raise ValueError(f"The maximum number of time steps to run must be given and must be a FINITE positive number: {max_time_steps}")
        # ---- UPDATE FOR CONTINUING TASK

        if use_fixed_average_reward and (estimated_average_reward is None or np.isnan(estimated_average_reward)):
            # Do NOT use a fixed average reward if the given average reward value is None or missing
            use_fixed_average_reward = False
            raise ValueError("The user requested to use a FIXED value of the average reward as correction of the value functions but the given average reward in `estimated_average_reward` is None or missing.")

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
        elif weights_rmse is None or not weights_rmse:
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

        # Only one of set_cycle and dict_proba_cycle must be given
        if set_cycle is not None and dict_proba_cycle is not None:
            raise ValueError(f"Only one out of parameters `set_cycle` and `dict_proba_cycle` can be given:\nset_cycle={set_cycle}\ndict_proba_cycle={dict_proba_cycle}")

        # Flag whether we are keeping track of cycles during the simulation
        keep_track_of_cycles = set_cycle is not None or dict_proba_cycle is not None

        # Setup the information needed when cycles are used to estimate the stationary distribution of states using renewal theory
        cycle_times = []  # Note that the first cycle time will include a time that may not be a cycle time because the cycle may have not initiated at the sytem's start state. So the first cycle time will be considered a delay time.
        num_cycles = 0
        last_cycle_entrance_time = 0       # We set the last cycle time (i.e. the moment when the system enters the cycle set) to 0 (even if it is unknown) so that we can easily compute the FIRST cycle time below as "t - last_cycle_entrance_time"
        expected_cycle_time = 0.0  # We set the expected cycle time so that we can compute the expected cycle time recursively
        # Array of state counts of EVERY state visited in COMPLETE cycles.
        # NOTE that:
        # - visits happening within the first "cycle" are EXCLUDED from these counts because the first cycle is ALWAYS incomplete, as the definition of a cycle START
        # requires knowledge of the current and the next state, and this information is NOT available at the beginning of the simulation.
        # - This array contains visit counts to ALL states visited within a cycle, not only of the state that is visited when ENTERING the cycle set, as may be suspected.
        # This can be used to estimate the stationary probability of visiting EACH environment state using its characterization based on renewal theory.
        state_counts_in_complete_cycles = np.zeros(self.env.getNumStates(), dtype=int)
        # Dictionaries that keep track of the counts of the states visited at every START cycle event and at every EXIT event from the cycle set (when set_cycle is not None)
        dict_state_counts_start_cycle = dict()      # This information can be used to estimate the stationary distribution of the state happening at the start of a cycle (useful in the context of FV with soft killing).
        dict_state_counts_end_cycle = dict()        # This information can be used to estimate the stationary distribution of the state happening just BEFORE the start of a cycle (useful in the context of FV with soft killing to select the start state of the FV particles so that they are OUTSIDE the set of states with high killing probability --i.e. of states with high visit frequency).
        dict_state_counts_exit_cycle_set = dict()   # This information can be used to estimate the stationary exit distribution from the cycle set.
        #--- Parse input parameters

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()

        # Set seeds of:
        # - torch --> responsible for selecting the action when the policy is modeled via a neural network.
        # - the policy's environment --> just in case this environment does NOT have the same memory address as self.env
        #   (e.g. when policies are deepcopied from the original policy in order to compare different learning methods)
        # - numpy --> responsible for deciding whether a (completely) random action is chosen (when epsilon_random_action > 0).
        #   numpy is used to both draw a random number to decide whether to choose a random action (without following the policy)
        #   and then to actually choose that random action, if this ends up being the case.
        # - the object's environment --> responsible of deciding on the next state given the action.
        if seed is not None:
            torch.manual_seed(seed)
            policy.env.seed(seed)
            np.random.seed(seed)
            self.env.setSeed(seed)

        # Reset the environment (this should be done BEFORE resetting the learner because the learner will most likely store the reward at the initial state by calling
        # self.env.getReward() on the self.env.getState(), and if the state stored in the environment (from e.g. a previous execution/replication of the learning process),
        # is NOT the state at which the environment will start after reset below
        # Optional start state JUST for the very first episode
        # (In case we want to start at a specific state and then perform the subsequent resets as per the environment initial state distribution)
        self.env.reset()
        if start_state_first_episode is not None:
            self.env.setState(start_state_first_episode)
        if show_messages(verbose, verbose_period, 0):
            print("@{}".format(get_current_datetime_as_string()))
            print(f"[t_learn={t_learn}] Agent starts at state {self.env.getState()} with reward {self.env.getReward(self.env.getState())}")
        if self.debug:
            print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

        # Reset the learner (i.e. prepare it for a fresh new learning experience with all learning memory erased and learning rates reset to their initial values)
        # Note that a special treatment may be granted to the reset of the value functions because we may NOT want to reset them,
        # for instance when we are learning a policy and we use this simulator to learn the value functions...
        # In that case, we don't want to start off at 0.0 again but to start off at the estimated values obtained
        # under the previous policy, which normally is very close to the new policy after one step of policy learning.
        # (In some situations --e.g. labyrinth where the policy is learned using Actor-Critic or policy gradient learning--
        # I observed (Nov-2023) that non-optimal policies are learned that minimize the loss function if we reset the
        # value functions to 0 at every policy learning step, while the problem does NOT happen when the value functions are NOT reset.)
        # Also, a separate strategy is considered for the reset (or not) of the average reward, as currently this is learned by using innovation information
        # that is different from the one used to learn the value functions, namely the "newly-observed-average-reward-in-episode" - "current-estimate-of-average-reward",
        # as opposed to the TD error that is used to learn the value functions.
        # In particular, when this is the very first policy learning step (t_learn=0 namely that a NEW experiment is run) or
        # when the simulation is run independently of a policy learning process (e.g. by unit tests where t_learn < 0) or
        # when no initially estimated average reward is given,
        # the average reward is reset to zero and learning starts again from scratch,
        # o.w. the given estimated average reward should be used as initial estimate of the average reward during further learning.
        learner.reset(reset_episode=True, reset_value_functions=reset_value_functions, reset_average_reward=t_learn <= 0 or (estimated_average_reward is None or np.isnan(estimated_average_reward)))
        print(f"[IN _run_single_continuing_task()] The average reward stored in learner after RESET is: {learner.average_reward}, {learner._average_reward_in_episode} (internal avg. reward attribute by EPISODE)")

        # Store initial values used in the analysis of all the episodes run
        V_state_observe, RMSE, MAPE, ntimes_rmse_inside_ci95 = self._initialize_run_with_learner_status(nepisodes, learner, compute_rmse, weights, state_observe)

        # Initial state value function
        if self.env.isStateContinuous():
            # We initialize DUMMY state value functions, so that we can refer to them as if the state space were discrete, so that we don't need to add many IF conditions below...
            # (2025/02/08) HOWEVER, maybe we should change this and call the self._get_state_value() and self._get_action_value() methods.
            V = np.repeat(0.0, self.env.getNumStates())
            Q = np.repeat(0.0, self.env.getNumStates()*self.env.getNumActions())
            A = np.repeat(0.0, self.env.getNumStates()*self.env.getNumActions())
        else:
            V = learner.getV().getValues()
            Q = learner.getQ().getValues()
            A = learner.getA().getValues()
        if DEBUG_ESTIMATORS:
            print("Value functions at start of experiment:")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))
            print("A = {}".format(A.reshape(self.env.getNumStates(), self.env.getNumActions())))

        # Check whether the state value function is ALL zeros, for all states
        # This is used when assigning a reward to states in EXIT events (i.e. reward_shaping=True)
        # in order to know whether we should assign the value of `reward_for_exit_states` or a value that is proportional to the state value function in absolute value.
        state_value_function_is_all_zeros = (np.min(V) == np.max(V) == 0.0)

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

        # Plotting setup
        if plot:
            # Setup the figures that will be updated at every verbose_period
            num_colors_in_colormap = max_time_steps
            fig_V, fig_V2, fig_C, fig_RMSE_state, colors_V, self.fig_policy = self._setup_plots(t_learn=t_learn, colormap=colormap, lut=num_colors_in_colormap, state_observe=state_observe)

            # Setup the axes to use for the average reward evolution plot (a new plot that is not created above by self._setup_plots())
            dict_axes = dict({'average_reward': plt.figure().subplots(1, 1)})
            # Initialize the average reward plot with the current estimate of the average reward stored in the learner
            dict_lines = dict({'average_reward': dict_axes['average_reward'].plot(0, learner.getAverageReward(), '.-', color="red")})
            # Set the maximum X axis value if the number simulation steps to run in advance is known
            dict_axes['average_reward'].set_xlim((None, max_time_steps)) if max_time_steps < +np.Inf else None
            dict_axes['average_reward'].set_xlabel("Step")
            dict_axes['average_reward'].set_ylabel("Estimated average reward")
            dict_axes['average_reward'].legend(["TD average reward"])
            plt.suptitle(f"[_run_single_continuing_task, Learning step {t_learn+1}]")

            # Reposition and resize the figure to avoid overlapping with the other figures
            # Ref: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
            plt.figure(dict_axes['average_reward'].get_figure().number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry( WINDOW_TOP_LEFT_HORIZONTAL,
                                        WINDOW_TOP_LEFT_VERTICAL + WINDOW_HEIGHT + 3*SPACE_BETWEEN_WINDOWS, # `3*` because we need to leave space for the WINDOW's title
                                        WINDOW_WIDTH,
                                        WINDOW_HEIGHT)

        if plot and keep_track_of_cycles and set_cycle is not None and reward_shaping: #False
            # Initialize the plot showing the absorption set and eventually the shaped rewards (when reward shaping is used)
            ax_absorption_and_rewards = plt.figure().subplots(1, 1)
            self.env.plot_points(list(set_cycle), ax=ax_absorption_and_rewards, style='x', markersize=5, color="red")
            _arr_rewards = self.env.getRewardsLandscape()
            self.env.plot_values(_arr_rewards, ax=ax_absorption_and_rewards, cmap="Greys", vmin=0, vmax=np.max(_arr_rewards))
            if reward_shaping:
                if np.isnan(reward_for_exit_states):
                    ax_absorption_and_rewards.set_title(f"Learning step t_learn = {t_learn+1}\nNO REWARD SHAPING AT THIS STEP")
                else:
                    ax_absorption_and_rewards.set_title(f"Learning step t_learn = {t_learn+1}\nDist. of ORIGINAL rewards")
            else:
                ax_absorption_and_rewards.set_title(f"Learning step t_learn = {t_learn+1}\nNo reward shaping")
            plt.pause(pause)
            plt.draw()

        # Iterate on the episodes to run
        nepisodes_max_steps_reached = 0
        max_time_steps_reached = False  # Flags whether the TOTAL number of steps reaches the maximum number of steps allowed over all episodes (so that we can break the FOR loop on episodes if that happens)
        t = 0   # Learning step counter: t represents the time at which the Markov chain transitions to the NEXT state. See more details at the @note at the beginning of the file.
                # Note that this time value is used for counting the number of steps / events and, when a cycle set has been given, to compute the expected cycle time,
                # (which can be used to estimate the stationary probability of states using renewal theory).
        episode = -1
        done = False
        while not done:
            episode += 1

            # Reset the environment (ONLY if it is not the very first step, because in that case, the reset has been already carried out above, before resetting the learner
            # --for the reasons described therein).
            # Note: This reset is typically carried out by the gym module, e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method
            # where the initial state is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state,
            # which is assumed to have been defined appropriately in order to have the start state the user wishes to use.
            if t > 0:
                self.env.reset()

            # Time step within the current episode (i.e. within the time period between the reset of the environment and reaching the terminal state).
            # Note that we initialize it at -1 because the time within an episode indexes the time at which the ACTION is taken,
            # (and this will happen AFTER increasing t_episode by 1, so that the first time an action is taken will be indexed by t_episode = 0)
            # as opposed to the time at which the next state is observed (which is the convention used for variable `t` defined above).
            # This is done like that so that we have the state-action-reward sequence S(0), A(0), R(1), S(1), A(1), R(2), ...
            # So, t_episode indexes the state BEFORE transitioning.
            t_episode = -1
            done_episode = self.env.getState() in self.env.getTerminalStates()
            stop = False
            while not stop:
                t += 1
                t_episode += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                # Store the current V(s) value as PREV for informational plotting purposes
                V_state_prev = self._get_state_value(learner, state)

                # ---- UPDATE FOR CONTINUING TASK
                if done_episode:
                    # The episode ended at the previous step
                    # => Reset the environment to a start state because the process honours a CONTINUING learning task (of the value functions)
                    # => Reset the episode-related information (needed most importantly for a correct calculation of the average reward)

                    if plot:
                        # Update plots that are updated at the end of an episode
                        self._update_plots_at_episode_end(episode, nepisodes, learner, t_learn, fig_V, fig_V2, fig_RMSE_state, colors_V, state_observe, ntimes_rmse_inside_ci95,
                                                          weights=weights, pause=pause, method_name="_run_single_continuing_task, ")

                        # Update the average reward plot, which happens only at the end of each "episode"
                        # (i.e. when the terminal state is reached or when the episode finishes because the maximum simulation time was reached)
                        # since the average reward is updated by the LeaTD.learn_at_episode_end() method.
                        dict_lines = update_plots(dict_axes, dict({'average_reward': [(t, learner.getAverageReward())]}), dict_lines, show_title=False)
                        # Make sure that the Y axis is re-scaled after adding the new point (because after setting the axis limits (as done right below with set_ylim()),
                        # the axis limit is NOT re-calculated after adding new points (see: https://matplotlib.org/stable/users/explain/axes/autoscale.html))
                        dict_axes['average_reward'].autoscale(axis='y')
                        if learner.getAverageReward() < 0:
                            # We assume the rewards are negative
                            # => Set the MAXIMUM at 0, so that we get a visual comparison with other figures plotting the same concept
                            dict_axes['average_reward'].set_ylim((None, 0))
                        else:
                            # We assume the rewards are positive
                            # => Set the minimum at 0, so that we get a visual comparison with other figures plotting the same concept
                            dict_axes['average_reward'].set_ylim((0, None))
                        plt.figure(dict_axes['average_reward'].get_figure().number)  # Need to select the figure so that it is re-drawn, otherwise, the redraw applies to the active figure (already redrawn) selected above by plt.figure()
                        plt.suptitle(f"[_run_single_continuing_task, Learning step {t_learn+1}]\nEpisode ended at episode time t_episode={t_episode}, simulation time t={t}"
                                     f"\n(More than one point is plotted ONLY when a terminal state has been reached at least once)")
                        plt.pause(pause)
                        plt.draw()

                        if isinstance(learner, LeaTDLambdaAdaptive):
                            learner.plot_info(episode, nepisodes)

                    # Reset the episode counter
                    # Note that we reset it to -1 and NOT 0 because the episode is considered to start when `state` is a START state and here state is a terminal state,
                    # whereas *`next_state`* is the start state. So t_episode = 0 should be set at the next iteration.
                    t_episode = -1

                    # Perform the action of going to an environment's start state
                    action = 0
                    next_state = self.env.reset()
                    reward = self.env.getReward(next_state)
                    done_episode = next_state in self.env.getTerminalStates()

                    # Reset the learner as a new episode will start
                    # It is important to reset all the episode-related information, most importantly the history of rewards observed in the episode,
                    # which is used to compute the average reward. If this reset is not done, most likely the average reward will be WAY underestimated
                    # because the update of the average reward function ((normally) Learner._update_average_reward()) will think that the number of rewards
                    # observed in the episode when updating the within-episode average reward is WAY larger than it really was.
                    learner.reset(reset_episode=False, reset_value_functions=False, reset_average_reward=False)

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
                    #
                    # NOTE that the `info` dictionary will be updated to its regular value (normally empty) at the next call to self.env.step() which happens at the next "regular"
                    # step of the agent (where the agent goes from a state to a VALID next state under the original EPISODIC Markov process
                    # --as opposed to going from a terminal state to a start state, which is what just happened above.
                    info = {'update_trajectory': False,
                            'update_counts': False}
                # ---- UPDATE FOR CONTINUING TASK
                else:
                    action = self._choose_action(policy, state, epsilon_random_action=epsilon_random_action)
                    next_state, reward, done_episode, info = self.env.step(action)
                if reward != 0.0:
                    print(f"*** NON ZERO REWARD [INI]!! (t={t}, state={state} ({self.env.getStateFromIndex(state, simulation=False) if not self.env.isStateContinuous() else state}), action={action}, next_state={next_state} ({self.env.getStateFromIndex(next_state, simulation=False) if not self.env.isStateContinuous() else next_state}), reward={reward})")

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
                        print("[DEBUG] (MAX TIME STEPS PER EPISODE = {} REACHED at episode{}!)".format(max_time_steps_per_episode, learner.episode))

                # Check end of simulation
                if t >= max_time_steps:
                    max_time_steps_reached = True
                    done_episode = True
                    stop = True
                    if self.debug:
                        print("[run_single_continuing_task, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, learner.episode))

                if self.debug:
                    # We place this message BEFORE calling learner.learn() below because we want to show the state value BEFORE and after updating it (by learner.learn())
                    print("t: {}, t in episode: {}, s={}, a={} -> ns={}, r={}: state_count={:.0f}, alpha={:.3f}, V({})={:.4f}, Q({},{})={:.4f} -> V({})=" \
                          .format(t, t_episode, state, action, next_state, reward, learner.getStateCounts()[self.env.getIndexFromState(state)], learner.getAlphaForState(self.env.getIndexFromState(state)),
                                  state, self._get_state_value(learner, state),
                                  state, action, self._get_action_value(learner, state, action),
                                  state), end="")

                if self.debug:
                    print("{:.4f}, Q=({},{})={:.4f}, avg. reward = {:.4f}".format(self._get_state_value(learner, state), state, action, self._get_action_value(learner, state, action), learner.getAverageReward()))
                if done_episode: #self.debug and done_episode:
                    print("--> [DEBUG] Done episode [{} iterations of {} per episode of {} total steps] at state {} with reward {}" \
                          .format(t_episode+1, max_time_steps_per_episode, max_time_steps, self.env.getState(), reward))

                # Observation state
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V_state_observe += [self._get_state_value(learner, state_observe)]

                # Check if the event that signals the START of a new cycle has occurred
                # This event depends on ONE of the following:
                # - the current and next states, when set_cycle is given.
                # - the simple occurrence of a clock signaling the event, when dict_proba_cycle is given.
                cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles, dict_state_counts_start_cycle, dict_state_counts_end_cycle = \
                    self._check_cycle_occurrence_and_update_cycle_variables(set_cycle, dict_proba_cycle, t, self.env.getIndexFromState(state), self.env.getIndexFromState(next_state),
                                                                            cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles,
                                                                            dict_state_counts_start_cycle, dict_state_counts_end_cycle)

                # Check if the system has EXITED the set of states defining a cycle
                _exit_state_has_not_yet_been_visited = next_state not in dict_state_counts_exit_cycle_set.keys()
                exit_event, dict_state_counts_exit_cycle_set = self._check_cycle_exit_and_update_exit_counts(set_cycle, self.env.getIndexFromState(state), self.env.getIndexFromState(next_state), dict_state_counts_exit_cycle_set)
                if exit_event and reward_shaping and not np.isnan(reward_for_exit_states) and not self.env.isTerminalState(next_state) and _exit_state_has_not_yet_been_visited:
                    # This is the first EXIT event at this next_state state and it does NOT happen at a terminal state (which is assumed to have a non-zero reward)
                    # => Set the reward of the exit state to the given reward_for_exit_states value and update the reward used below to learn value functions and the long-run expected reward
                    # NOTE that the shaped reward is chosen equal to the state value function |V(s)|, unless all values of V(s) (for all states) are zero, in which case we use the given constant value of `reward_for_exit_states`
                    reward = reward_for_exit_states if state_value_function_is_all_zeros else np.abs(self._get_state_value(learner, next_state))
                    self.env.setReward(next_state, reward)
                    print(f"[run_single_continuing_task] REWARD SHAPING: r={reward} assigned to exit state {next_state} ({self.env.getStateFromIndex(next_state, simulation=False) if not self.env.isStateContinuous() else next_state})")

                # Learn: i.e. update the value functions (stored in the learner) for the *currently visited state and action* with the new observation
                # NOTE: If this call to learn() is done at the end of the episode (done_episode = True) further updates of information is performed
                # (e.g. update of the average reward stored in GenericLearner (its average_reward attribute) via the LeaTD.learn_at_episode_end() method when the learner is TD)
                if use_fixed_average_reward:
                    # Always use the same (fixed) average reward value as correction of the value functions at every learning step
                    # Note that the check of whether estimated_average_reward is missing has been done at the beginning when parsing input parameters,
                    # in which case parameter use_fixed_average_reward is set to False.
                    info['average_reward'] = estimated_average_reward
                learner.learn(t_episode, state, action, next_state, reward, done_episode, info)
                if state in self.env.getTerminalStates():
                    # We need to copy the Q-values of the terminal state to the other actions because the action chosen to go to the start state is always the same (action 0)
                    action_anchor = 0
                    for _action in range(self.env.getNumActions()):
                        learner.getQ()._setWeight(state, _action, learner.getQ().getValue(state, action_anchor))
                        learner.getA()._setWeight(state, _action, learner.getA().getValue(state, action_anchor))

                #---- UPDATE FOR CONTINUING TASK
                # Plotting step moved INSIDE the episode because there is only 1 episode!
                if plot and show_messages(True, verbose_period, t_episode):
                    self._update_plots(learner, t_learn, fig_V, fig_C, colors_V, num_colors_in_colormap, t, max_time_steps, pause=pause, method_name="_run_single_continuing_task, ")
                #---- UPDATE FOR CONTINUING TASK

                if show_messages(verbose, verbose_period, t) or show_messages(True, max_time_steps // 10, t):
                    # Inform about progress at every verbose_period or at every 10% of steps (this is what the `// 10` of the second call to show_messages() means)
                    print("[value functions, {:.0f}%] Completed time step {} out of {}".format(t/max_time_steps*100, t, max_time_steps))

            #------- FICTITIOUS EPISODE FINISHED --------#
            print(", agent ENDS at state: {} at discrete time t = {})".format(self.env.getState(), t), end=" ")
            print("")

            if not self.env.isStateContinuous():
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
                        ref_V_true, ref_V = self._compute_function_reference_values()
                        if weights_rmse is not None:
                            weights = learner.getStateCounts()
                        RMSE[min(episode+1, nepisodes)] = rmse(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)
                        MAPE[min(episode+1, nepisodes)] = mape(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)

            # Use the following IF when we want to stop the simulation EITHER when the maximum time steps (over all episodes) is reached
            # OR
            # when the number of episodes has been run.
            if max_time_steps_reached or episode == nepisodes - 1:
                # The maximum number of steps over ALL episodes has been reached, so we need to stop the simulation
                done = True

        if plot and reward_shaping and not np.isnan(reward_for_exit_states):
            # Plot the distribution of the shaped rewards
            _arr_rewards = self.env.getRewardsLandscape()
            self.env.plot_values(_arr_rewards, ax=ax_absorption_and_rewards, cmap="Greys", vmin=0, vmax=np.max(_arr_rewards), add_colorbar=False)
            ax_absorption_and_rewards.set_title(f"Learning step t_learn = {t_learn+1}\nDist. of SHAPED rewards and true rewards")
            plt.pause(pause)
            plt.draw()

        # Reset the reward to ZERO for the states affected by the reward shaping (if any) to promote the visit of EXIT states from A.
        # This is important in the context of Fleming-Viot learning because reward shaping should only take place during the initial exploration of the environment,
        # o.w. we are changing the environment itself on which an optimal policy is learned. Reward shaping should only be used to try a policy other than the one that has been
        # repetitively used without success, to observe EXIT states from the absorption set A that allow the execution of the FV simulation.
        # *** IMPORTANT ASSUMPTION: Only terminal states have a non-zero reward. ***
        if reward_shaping and not np.isnan(reward_for_exit_states):
            for s in dict_state_counts_exit_cycle_set.keys():
                if not self.env.isTerminalState(s):
                    self.env.setReward(s, 0.0)

        # Compute the stationary distribution of the state at the start of a cycle and of the state when exiting the cycle set (which is considered to be OUTSIDE of the cycle set)
        # These variables are dictionaries that are never None (when no cycle set has been specified, their values are empty dictionaries, as deduced from the function called here)
        probas_stationary_start_cycle = compute_proba_distribution_from_counts(dict_state_counts_start_cycle)
        probas_stationary_end_cycle = compute_proba_distribution_from_counts(dict_state_counts_end_cycle)
        probas_stationary_exit_cycle_set = compute_proba_distribution_from_counts(dict_state_counts_exit_cycle_set, set_cycle=set_cycle)
            ## For the EXIT state distribution, note that ONLY the states in the outside boundary (a.k.a. "activation set" in the FV context)
            ## that are VISITED by the excursion are updated.
            ## This may NOT be the best option because there may be states that are never visited and they will NOT be added to the exit distribution (with probability 0)...
            ## At this point (2024/12/22) I am not sure if this is a problem... although on 2024/03/30 I wrote the following to-do task:
            ## "Consider ALL states in the outside boundary of the cycle set in the estimation of the exit distribution (to solve the problem written just above).
            ## To implement this, the information about the exit states set should be given in a separate parameter, as the simulation implemented in the current method knows nothing about FV."
            ## SO, STILL TO CLARIFY IF THIS IS AN IMPORTANT TO-DO TASK...

        if True or DEBUG_ESTIMATORS:
            if set_cycle is not None and len(set_cycle) > 0:
                print(f"Estimated distribution of EXIT states from the cycle set (on {num_cycles} cycles):")
                for idx_state in sorted(probas_stationary_exit_cycle_set.keys()):
                    print(f"{idx_state}: {probas_stationary_exit_cycle_set[idx_state]}")
                print("")
            if dict_proba_cycle is not None:
                print(f"Estimated distribution of END-cycle states (on {num_cycles} cycles):")
                for idx_state in sorted(probas_stationary_end_cycle.keys()):
                    print(f"{idx_state}: {probas_stationary_end_cycle[idx_state]}")
                print("")

            if self.env.isStateContinuous():
                V = np.repeat(0.0, self.env.getNumStates())
                Q = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
                A = np.repeat(0.0, self.env.getNumStates() * self.env.getNumActions())
            else:
                V = learner.getV().getValues()
                Q = learner.getQ().getValues()
                A = learner.getA().getValues()
            print(f"Estimated value functions at END of experiment (last episode run = {learner.episode}):")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))
            print("A = {}".format(A.reshape(self.env.getNumStates(), self.env.getNumActions())))

        if plot:
            self._final_plots(learner, t_learn, fig_V, fig_C, method_name="_run_single_continuing_task, ")

        return  learner.getV().getValues(), learner.getQ().getValues(), learner.getA().getValues(), learner.getStateCounts(), RMSE, MAPE, \
                {   'nsteps': t,
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
                    'num_cycles': num_cycles if keep_track_of_cycles else None,
                    'last_cycle_entrance_time': last_cycle_entrance_time if keep_track_of_cycles else None,
                    'expected_cycle_time': expected_cycle_time if keep_track_of_cycles else None,
                    'state_counts_in_complete_cycles': state_counts_in_complete_cycles if keep_track_of_cycles else None,
                    'probas_stationary_start_cycle': probas_stationary_start_cycle,
                    'probas_stationary_end_cycle': probas_stationary_end_cycle,
                    'probas_stationary_exit_cycle_set': probas_stationary_exit_cycle_set,
                }

    def _get_state_value(self, learner, state):
        "Returns the state value for the given state stored in the given learner depending on whether the function is tabular or not and on whether the state is continuous or not"
        return learner.getV().getValue(self.env.getIndexFromState(state)) if self.env.isStateContinuous() and learner.getV().isTabular() else learner.getV().getValue(state)

    def _get_action_value(self, learner, state, action):
        "Returns the action value for the given state and action stored in the given learner depending on whether the function is tabular or not and on whether the state is continuous or not"
        return learner.getQ().getValue(self.env.getIndexFromState(state), action) if self.env.isStateContinuous() and learner.getQ().isTabular() else learner.getQ().getValue(state, action)

    def _get_advantage_value(self, learner, state, action):
        "Returns the action value for the given state and action stored in the given learner depending on whether the function is tabular or not and on whether the state is continuous or not"
        return learner.getA().getValue(self.env.getIndexFromState(state), action) if self.env.isStateContinuous() and learner.getA().isTabular() else learner.getA().getValue(state, action)

    def _choose_action(self, policy, state, epsilon_random_action=0.0):
        """
        Chooses an action to take on the environment (stored in the object) at the given state, following the given policy, with the given probability of taking a random action

        A positive probability of choosing a random action can be given in order to have ergodic Markov chains under deterministic policies or under
        policies having a subset of the actions with probability zero.
        """
        if epsilon_random_action == 0 or np.random.uniform() > epsilon_random_action:
            # Choose an action following the policy if no random action should be taken
            action = policy.choose_action(state)
        else:
            # Choose a random action with probability `epsilon_random_action`
            action = np.random.choice(np.arange(self.env.getNumActions()))

        return action

    @staticmethod
    def _check_cycle_occurrence_and_update_cycle_variables(set_cycle, dict_proba_cycle, t, state, next_state,
                                                           cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles,
                                                           dict_state_counts_start_cycle, dict_state_counts_end_cycle):
        """
        Checks the occurrence of a cycle and updates the estimated cycle time and related tracking variables (see the description of `Return`)

        Arguments:
        set_cycle: set or None
            Set whose visit from a state NOT in `set_cycle` defines a new cycle whenever an entrance event to the set has been observed previously.
            It must be None if dict_proba_cycle is not None.

        dict_proba_cycle: dict or None
            Dictionary with the probability of starting a new cycle at each environment state.
            This assumes that the environment state space is finite, as the state indexes the dictionary.
            The probability of the `next_state` (NOT of `state`) is used to check whether a new cycle event occurs. There are two reasons for this logic:
            - at the very beginning of the simulation, the environment starts at a given state but at that moment we cannot consider that a new cycle event
            occurs.
            - the cycle event is an ABSORPTION event when running the FV process, which requires that the particle changes state. Given that change,
            and under the traditional approach of defining an absorption set A, the absorption condition is checked on whether the NEXT state belongs
            to the absorption set NOT whether the *current* state belongs to the absorption set A.
            It must be None if set_cycle is not None.

        t: positive float
            Time elapsed since the start of the process simulation.
            Note that this time should NOT be reset by the completion of an episode.

        state, next_state: int
            Indices of the current and next state on which the check of whether the process enters a cycle state is done, in which case a new cycle is
            considered to have started.

        cycle_times: list
            List containing the observed cycle times, including the first cycle, even if it is an incomplete cycle because it doesn't measure the time
            between two consecutive entrances to the cycle set or between two consecutive cycle starts (because a cycle start requires analyzing two states,
            the current and the next state, and this information is NOT available at the very beginning of the simulation).

        state_counts_in_complete_cycles: numpy.ndarray
            Array containing the count of EACH state visited during a cycle. A state visit is counted ONLY when it happens within COMPLETE cycles,
            i.e. the visits happening in the first cycle (which is an incomplete cycle --see `cycle_times` description) do not contribute to these statistics.
            This information can be used, for instance, to estimate the stationary state probability distribution using its characterization based on
            renewal theory.

        last_cycle_entrance_time: positive float
            Absolute time of the start of the previous cycle. This is used to define whether the occurrence of the event signalling a new cycle
            closes a COMPLETE cycle: when last_cycle_entrance_time = 0, it means that this is NOT the case, o.w. the closed cycle is complete.

        expected_cycle_time: positive float
            Estimate of the expected cycle time based on all the observed COMPLETE cycle times.

        num_cycles: int
            Number of observed COMPLETE cycles.

        dict_state_counts_start_cycle: dict
            Dictionary indexed by the state containing the visit count of the states at which new cycles are observed to start.

        dict_state_counts_end_cycle: dict
            Dictionary indexed by the state containing the visit count of the states visited just BEFORE new cycles start.

        Return: tuple
        Tuple containing UPDATED values for the following quantities:
        - cycle_times: number of cycle times observed, including the first INCOMPLETE cycle.
        - state_counts_in_complete_cycles: list indexed by the state index containing the state visit count for each state visited within COMPLETE cycles.
        - last_cycle_entrance_time: absolute time of the start of the previous cycle.
        - expected_cycle_time: estimate of the expected cycle time.
        - num_cycles: number of complete cycles observed.
        - dict_state_counts_start_cycle: dictionary containing the counts of each state at which a new cycle starts. This is useful when e.g. estimating
        the distribution of the cycle start state in the context of FV with soft killing.
        - dict_state_counts_end_cycle: dictionary containing the counts of each state visited just BEFORE a new cycle starts.
        This is useful when e.g. estimating the distribution of the state just before a cycle starts in the context of FV with soft killing,
        in which case it could be used to select the start state of the FV particle system (so that particles start OUTSIDE the set of states with
        high killing probability).
        """
        #--- Parse input parameters
        # Only one of set_cycle or dict_proba_cycle can be given
        if set_cycle is not None and dict_proba_cycle is not None:
            raise ValueError(f"Only one out of parameters `set_cycle` and `dict_proba_cycle` can be given:\nset_cycle={set_cycle}\ndict_proba_cycle={dict_proba_cycle}")
        #--- Parse input parameters

        # Functions defining the start of a new cycle
        entered_set_cycle = lambda s, ns: s not in set_cycle and ns in set_cycle
        event_cycle_occurred = lambda s: np.random.uniform() <= dict_proba_cycle.get(s, 0.0)

        if set_cycle is not None or dict_proba_cycle is not None:
            if set_cycle is not None and entered_set_cycle(state, next_state) or \
               dict_proba_cycle is not None and event_cycle_occurred(next_state):
                # Update the count of the state at which the cycle ends and at which the cycle starts
                dict_state_counts_end_cycle[state] = dict_state_counts_end_cycle.get(state, 0) + 1
                dict_state_counts_start_cycle[next_state] = dict_state_counts_start_cycle.get(next_state, 0) + 1

                # Note on the cycle time calculation:
                # The fact that we use `t` to compute the cycle time --instead of `t_episode`-- indicates that we are considering the task to be a continuing learning task,
                # as opposed to an episodic learning task, because the value of `t` is NOT reset at the beginning of each episode,
                # instead it keeps increasing with every new time step.
                cycle_times += [t - last_cycle_entrance_time]
                # We mark the time the system entered the cycle set, so that:
                # - we can compute the next cycle time
                # - we can decide when to start measuring cycle times because we know they correspond to FULL cycle times.
                last_cycle_entrance_time = t
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
                # The process entered the cycle set at least once
                state_counts_in_complete_cycles[state] += 1

        return cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles, dict_state_counts_start_cycle, dict_state_counts_end_cycle

    @staticmethod
    def _check_cycle_exit_and_update_exit_counts(set_cycle, state, next_state, dict_state_counts_exit_cycle_set):
        """
        Checks the occurrence of a cycle and updates the estimated cycle time and related tracking variables

        Arguments:
        set_cycle: set
            Set whose visit from a state NOT in `set_cycle` defines a new cycle whenever an entrance event to the set has been observed previously.

        state: int
            Index of the state before the current transition.

        next_state: int
            Index of the state after the current transition.

        dict_state_counts_exit_cycle_set: dict
            Dictionary containing the exit count of each state in the cycle set boundary.
            Initially the dictionary can be empty and the key associated to an exit state will be either created or updated, if already existing in the dict.
            The states in either side of the boundary are updated, i.e. the state in the inside boundary and the state in the outside boundary,
            respectively `state` and `next_state` whenever an exit event occurs.

        Return: tuple
        Duple with two elements:
        - Whether the transition from `state` to `next_state` corresponds to a cycle exit event.
        - The updated dict_state_counts_exit_cycle_set dictionary, based on the current visit of `state` and `next_state`.
        """
        exited_set_cycle = lambda s, ns: s in set_cycle and ns not in set_cycle
        exit_event = False
        if set_cycle is not None:
            exit_event = exited_set_cycle(state, next_state)
            if exit_event:
                # EXIT event because next_state is OUTSIDE the cycle set
                # Note that both the state INSIDE the cycle set (state) and the state OUTSIDE the cycle set (next_state) are updated
                # so that we keep track of BOTH two states through which the system exits the cycle set.
                # Based on the information of what states are part of the cycle set and what states are not we can further filter what counts we want to look at.
                dict_state_counts_exit_cycle_set[state] = dict_state_counts_exit_cycle_set.get(state, 0) + 1
                dict_state_counts_exit_cycle_set[next_state] = dict_state_counts_exit_cycle_set.get(next_state, 0) + 1

        return exit_event, dict_state_counts_exit_cycle_set

    def _initialize_run_with_learner_status(self, nepisodes, learner, compute_rmse, weights, state_observe):
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
            ref_V_true, ref_V = self._compute_function_reference_values()
            print(f"True value function:\n{self.env.getV() - ref_V_true}")
            RMSE[0] = rmse(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)
            MAPE[0] = mape(self.env.getV() - ref_V_true, learner.getV().getValues() - ref_V, weights=weights)
        else:
            RMSE = None
            MAPE = None
        if state_observe is not None:
            V = [self._get_state_value(learner, state_observe)]
        else:
            V = None
        # Keep track of the number of times the true value function of the state (assumed 0!)
        # is inside its confidence interval. In practice this only works for the mid state
        # of the 1D gridworld with the random walk policy.
        ntimes_rmse_inside_ci95 = 0

        return V, RMSE, MAPE, ntimes_rmse_inside_ci95

    def _compute_function_reference_values(self):
        """
        Computes reference values for the state value functions (true and estimated) to use in plots

        This is relevant for the AVERAGE learning criterion (typically used in CONTINUING learning tasks),
        as in that case the state value function is NOT unique (by linear algebra considerations).

        For the DISCOUNTED learning criterion, the reference values are set to 0.

        All the information is taken from the attributes of this object. If the environment does NOT have any true state value function stored in it,
        the reference value is set to 0.

        Return: Tuple
        Tuple with the following two elements:
        - Reference value for the true V(s), computed as the average of "true V(s)".
        - Reference value for the estimated V(s), computed as the average of "estimated V(s)".
        """
        ref_V_true = ref_V = 0.0
        if self.agent.getLearner().getLearningCriterion() == LearningCriterion.AVERAGE:
            ref_V_true = np.nanmean(self.env.getV()) if self.env.getV() is not None else 0.0
            ref_V = np.nanmean(self.agent.getLearner().getV().getValues())
        return ref_V_true, ref_V

    def _setup_plots(self, t_learn=0, colormap="seismic", lut=None, state_observe=None, setup_policy_plot=True):
        """
        Sets up the plots when iterative plots are requested

        These mainly includes the plot of the state value function V(s) estimated during the learning process, the state counts, the policy,
        and possibly the RMSE of a particle state of interest.

        t_learn: (opt) int
            Policy learning step at which this method is called (for informational purposes).
            default: 0

        colormap: (opt) str
            Name of the colormap to use in the generation of the animated plots showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.
            default: seismic, a colormap that ranges from blue to red, where the middle is white

        lut: (opt) int
            Lookup Up Table size defining the number of colors in the colormap used for the value function plots.
            The colormap is defined by the matplotlib.cm.get_cmap() function.
            default: None

        setup_policy_plot: (opt) bool
            Whether to setup (create) a figure for the plot of the policy when the value functions learning process ends.
            default: True

        Return: tuple
        Tuple with the following elements:
        - fig_V: figure handle for the line plot of the state value function. It is NEVER None.
        - fig_V2: (It is None for 1D environments) figure handle for the image plot of the state value function for 2D environments.
        - fig_C: (It is None for 1D environments as state counts are plotted on fig_V) figure handle for the image plot of the state count for 2D environments.
        - fig_RMSE_state: (It is None when state_observe=None) figure handle for the RMSE of V(s) for the state s to observe, if any.
        - colors_V: colormap object generated by cm.get_cmap() to use for the state value function plots. A color in the map can be retrieved by color(i/lut) where
        i is an integer between 0 and lut-1 defining the color number in the colormap we want to retrieve.
        - fig_P: (It is None when setup_policy_plot=False) figure handle for the image plot of the policy at each state.
        """
        # Colormap for the plot of the state value function V(s)
        colors_V = cm.get_cmap(colormap, lut=lut)

        # 1D plot (even for 2D environments)
        fig_V = plt.figure()
        ax = plt.gca()
        # Plot the true state value function (to have it as a reference already) and set integer values on the horizontal axis as states are integer-valued
        _ref_V_true, _ = self._compute_function_reference_values()
        if self.env.getV() is not None:
            ax.plot(self.env.getAllStates(), self.env.getV() - _ref_V_true, '.-', color="blue")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Plot the initial estimate of the state value function stored in the learner
        _learner_state_values = self.agent.getLearner().getV().getValues()
        ax.plot(self.env.getAllStates(), _learner_state_values - _ref_V_true, '-', color=colors_V(0))

        # Create other figures, depending on the environment's dimension
        if self.env.getDimension() == 2:
            # 2D plots
            fig_V2 = plt.figure()   # 2D state value function and 2D state counts
            fig_C = plt.figure()    # State counts
        else:
            # Add a secondary Y axis to fig_V where we will show the state counts
            fig_V.gca().twinx()
            fig_V2 = None
            fig_C = None

        # Plot for the state to observe
        if state_observe is not None:
            # Plot with the evolution of the V estimate and its error
            fig_RMSE_state = plt.figure()
        else:
            fig_RMSE_state = None

        if setup_policy_plot:
            # Image plot of the policy, it has as many subplots as state in the environment, laid out with its shape
            fig_P = plt.figure()
            # Plot the initial policy
            self._update_policy_plot(fig_P, title_prefix=f"[INITIAL, Learning step {t_learn+1}]\n")
        else:
            fig_P = None

        # Reposition and resize the figures so that we can see their updated clearly, without overlapping
        # It's NOT so easy!! The mechanism also depends on the matplotlib backend! (e.g. Qt5Agg or another one)
        # Ref: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
        # The method window.setGeometry() is used which modifies the position and size of the ACTIVE figure.
        # Its parameters are:
        # - position of window's top left corner, horizontal position from left
        # - position of window's top left corner, vertical position from top
        # - window width in pixels
        # - window height in pixels
        # ALSO, we need to leave a margin for the figure title, which is NOT accounted for in the window size!
        plt.figure(fig_V.number)
        fig_mgr = plt.get_current_fig_manager()
        fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL,
                                   WINDOW_TOP_LEFT_VERTICAL,
                                   WINDOW_WIDTH,
                                   WINDOW_HEIGHT)
        if fig_V2 is not None:
            plt.figure(fig_V2.number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS,
                                       WINDOW_TOP_LEFT_VERTICAL,
                                       WINDOW_WIDTH,
                                       WINDOW_HEIGHT)
        if fig_C is not None:
            plt.figure(fig_C.number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + 2*(WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS),
                                       WINDOW_TOP_LEFT_VERTICAL,
                                       WINDOW_WIDTH,
                                       WINDOW_HEIGHT)
        if fig_RMSE_state is not None:
            plt.figure(fig_RMSE_state.number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + 3*(WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS),
                                       WINDOW_TOP_LEFT_VERTICAL,
                                       WINDOW_WIDTH,
                                       WINDOW_HEIGHT)
        if fig_P is not None:
            plt.figure(fig_P.number)
            fig_mgr = plt.get_current_fig_manager()
            fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + 2*(WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS),
                                       WINDOW_TOP_LEFT_VERTICAL + WINDOW_HEIGHT + 3*SPACE_BETWEEN_WINDOWS,  # `3*` because we need to leave space for the WINDOW's title
                                       WINDOW_WIDTH,
                                       WINDOW_HEIGHT)

        return fig_V, fig_V2, fig_C, fig_RMSE_state, colors_V, fig_P

    def _update_plots(self, learner, t_learn, fig_V, fig_C, colors_V, colors_V_length, t, max_time_steps,
                      points_to_add_in_counts_plot=None,  # These are points to add to the fig_C plot given as 1D state index format, e.g. the 1D state indices of the absorption set
                      pause=0.1, method_name="", fontsize_labels=LABELS_FONTSIZE, color_labels=LABELS_COLOR):
        # NOTE: If fig_C is None and a state counts plot should be generated SEPARATE from the state value function plot (because e.g. the environment is 2D),
        # a new figure is created by this method. This is useful when we want to see how the state counts evolve.

        # Update the state value function plot
        plt.figure(fig_V.number)
        ax = plt.gca()

        _, ref_V = self._compute_function_reference_values()
        ax.plot(self.env.getAllStates(), learner.getV().getValues() - ref_V, linewidth=0.5, color=colors_V(t / colors_V_length))
        ax.set_xlabel(f"1D state index (0 - {self.env.getNumStates()-1})")
        ax.set_ylabel(f"V(s) - {ref_V}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(f"[{method_name}UPDATE, Learning step {t_learn+1}]\nV(s) evolution (blue: true, blueish: initial, reddish: final): # Steps={t} of {max_time_steps}"
                  f"\nAverage[V(s) - avg.R] = {np.mean(learner.getV().getValues() - ref_V):0.4f}", fontsize=10)
                  #f"\nSystem's time t = {t} of max {max_time_steps}, from state = {state} --> next state = {next_state} (V(s) = {np.round(self._get_state_value(learner, next_state), 3)})"
                  #f"\nV(state={state}) = {V_state_prev is not None and np.round(V_state_prev, 3) or 'N/A'} --> V(state={state}) = {np.round(self._get_state_value(learner, state), 3)} (delta(V) = {V_state_prev is not None and np.round(self._get_state_value(learner, state) - V_state_prev, 3) or 'N/A'}, {V_state_prev is not None and np.round((self._get_state_value(learner, state) - V_state_prev) / max(1, V_state_prev) * 100, 1) or 'N/A'}%)"
                  #f"\nCount[s] = {learner.getStateCounts()}, alpha[s] = {learner.getAlphasByState()}")
        plt.pause(pause)
        plt.draw()

        # Update of the state counts plot (or creation of a new one if environment is 2D and fig_C is None)
        if self.env.getDimension() == 1:
            # Add the state counts to the plot of the state value function
            axes = fig_V.get_axes()
            assert len(axes) == 2
            axes[1].cla()   # Clear the plot in the axis so that the color of the bars do not get more and more intense which would cover the plot of the state value function
            axes[1].bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.sca(axes[0])  # Go back to the primary axis
        else:
            # Plot the state counts as a separate image from the image containing the state value function (because the environment is 2D)
            if fig_C is None:
                fig_C = plt.figure()
                fig_mgr = plt.get_current_fig_manager()
                fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + 2*(WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS),
                                           WINDOW_TOP_LEFT_VERTICAL,
                                           WINDOW_WIDTH,
                                           WINDOW_HEIGHT)
            plt.figure(fig_C.number)
            plt.clf()   # Clear the figure in order to properly see the labels showing the state counts (o.w. they overlap on the existing ones when the figure they are plotted on is not new, i.e. when the value of the fig_C parameter is not None)
            ax_C = plt.gca()

            state_counts_min = np.min(learner.getStateCounts())
            state_counts_mean = np.mean(learner.getStateCounts())
            state_counts_max = np.max(learner.getStateCounts())

            arr_state_counts = learner.getStateCounts()
            self.env.plot_values(arr_state_counts, ax=ax_C, cmap="Blues", vmin=0, vmax=np.max(arr_state_counts))
            if points_to_add_in_counts_plot is not None:
                self.env.plot_points(points_to_add_in_counts_plot, ax=ax_C, style='x', markersize=5, color="red")
            self._add_count_labels(ax_C, arr_state_counts, fontsize=fontsize_labels, color=color_labels)

            plt.title("[{}UPDATE, Learning step {}]\nState visit counts: # visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                      .format(method_name, t_learn+1, state_counts_min, state_counts_mean, state_counts_max))
            plt.pause(pause)
            plt.draw()

    def _update_plots_at_episode_end(self, episode, nepisodes, learner, t_learn, fig_V, fig_V2, fig_RMSE_state, colors_V, state_observe, ntimes_rmse_inside_ci95,
                                     weights=None, pause=0.1, method_name="", fontsize_labels=LABELS_FONTSIZE, color_labels=LABELS_COLOR):
        # Plot the estimated state value function at the end of the episode, in both 1D layout and 2D layout, if the environment is 2D.
        #print("episode: {} (T={}), color: {}".format(episode, t_episode+1, colors_V(episode/nepisodes)))
        plt.figure(fig_V.number)
        _, ref_V = self._compute_function_reference_values()
        plt.plot(self.env.getAllStates(), learner.getV().getValues() - ref_V, linewidth=0.5, color=colors_V(min(episode, nepisodes-1) / nepisodes))
        plt.gca().set_xlabel(f"1D state index (0 - {self.env.getNumStates()-1})")
        plt.gca().set_ylabel(f"V(s) - {ref_V}")
        plt.title(f"[{method_name}UPDATE@Episode end, Learning step {t_learn+1}]\nV(s) evolution (blue: true, blueish: initial, reddish: final): Episode {episode+1} of {nepisodes}"
                  f"\nAverage[V(s) - avg.R] = {np.mean(learner.getV().getValues() - ref_V):0.4f}", fontsize=10)
        plt.pause(pause)
        plt.draw()
        # fig_V.canvas.draw()    # This should be equivalent to plt.draw()
        if fig_V2 is not None:
            # Update the 2D plots
            plt.figure(fig_V2.number)
            (ax_V2, ax_C) = fig_V2.subplots(1, 2)   # Layout is LEFT: state value function, RIGHT: state counts

            arr_state_values = np.asarray(learner.getV().getValues())
            self.env.plot_values(arr_state_values, ax=ax_V2, cmap=colors_V)
            ax_V2.set_title("State values, V(s)")

            arr_state_counts = learner.getStateCounts()
            ax_C.cla()  # Clear the plot to avoid overlap with new plot
            self.env.plot_values(arr_state_counts, ax=ax_C, cmap="Blues", vmin=0, vmax=np.max(arr_state_counts))
            self._add_count_labels(ax_C, arr_state_counts, fontsize=fontsize_labels, color=color_labels, factor_fontsize=0.8)
            ax_C.set_title("State visit count")

            fig_V2.suptitle(f"[{method_name}UPDATE@Episode, Learning step {t_learn+1}]\nEpisode {episode+1} of {nepisodes}")
            plt.pause(pause)
            plt.draw()

        if state_observe is not None:
            assert fig_RMSE_state is not None
            plt.figure(fig_RMSE_state.number)
            ax = plt.gca()

            # Compute quantities to plot
            ref_V_true, ref_V = self._compute_function_reference_values()
            RMSE_state_observe = rmse(np.array(self.env.getV()[state_observe] - ref_V_true), np.array(self._get_state_value(learner, state_observe) - ref_V), weights=weights)
            se95 = 2 * np.sqrt(0.5 * (1 - 0.5) / (episode + 1))
            # Only count falling inside the CI starting at episode 100 so that
            # the normal approximation of the SE is more correct.
            if episode + 1 >= 100:
                ntimes_rmse_inside_ci95 += (RMSE_state_observe <= se95)

            # Plot of the estimated value of the state at the end of the episode
            # NOTE: the plot is just ONE point, the state value at the end of the episode;
            # at each new episode, a new point with this information will be added to the plot.
            ax.plot(episode + 1, self._get_state_value(learner, state_observe) - ref_V, 'r*-', markersize=7)
            # Plot the average of the state value function over the values that it took along the episode while it was being learned.
            # plt.plot(episode, np.mean(np.array(V_state_observe)), 'r.-')
            # Plot of the estimation error and the decay of the "confidence bands" for the true value function
            ax.plot(episode + 1, RMSE_state_observe, 'k.-', markersize=7)
            ax.plot(episode + 1, se95, color="gray", marker=".", markersize=5)
            ax.plot(episode + 1, -se95, color="gray", marker=".", markersize=5)
            # Plot of learning rate
            # plt.plot(episode+1, learner.getAlphaForState(state_observe), 'g.-')

            # Finalize plot
            ax.set_ylim((-1, 1))
            yticks = np.arange(-10, 10) / 10
            ax.set_yticks(yticks)
            ax.axhline(y=self.env.getV()[state_observe], color="red", linewidth=0.5)
            ax.axhline(y=0, color="gray")
            legend = ['Estimated value', '|error|', 'true value', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)']
            ax.legend(legend)
            if episode + 1 == nepisodes:
                # Show the true value function coverage as x-axis label
                ax.set_xlabel("% Episodes error is inside 95% Confidence Interval (+/- 2*SE) (for episode>=100): {:.1f}%" \
                              .format(ntimes_rmse_inside_ci95 / (episode + 1 - 100 + 1) * 100))
            ax.legend(legend)
            plt.title(f"[{method_name}UPDATE@Episode, Learning step {t_learn+1}]\nState value (estimated and true) and |error| for state {state_observe} - episode {episode+1} of {nepisodes}")
            plt.pause(pause)
            plt.draw()

    def _final_plots(self, learner, t_learn, fig_V, fig_C,
                     points_to_add_in_counts_plot=None,  # These are points to add to the fig_C plot given as 1D state index format, e.g. the 1D state indices of the absorption set
                     method_name="", fontsize_labels=LABELS_FONTSIZE, color_labels=LABELS_COLOR):
        if self.env.getDimension() == 2:
            # The state counts plot is shown on a separate image than the state value function plot
            if fig_C is None:
                fig_C = plt.figure()
                fig_mgr = plt.get_current_fig_manager()
                fig_mgr.window.setGeometry(WINDOW_TOP_LEFT_HORIZONTAL + 2*(WINDOW_WIDTH + SPACE_BETWEEN_WINDOWS),
                                           WINDOW_TOP_LEFT_VERTICAL,
                                           WINDOW_WIDTH,
                                           WINDOW_HEIGHT)
            plt.figure(fig_C.number)
            plt.clf()   # Clear the figure in order to properly see the labels showing the state counts (o.w. they overlap on the existing ones when the figure they are plotted on is not new, i.e. when the value of the fig_C parameter is not None)
            ax_C = plt.gca()

            state_counts_min = np.min(learner.getStateCounts())
            state_counts_mean = np.mean(learner.getStateCounts())
            state_counts_max = np.max(learner.getStateCounts())

            arr_state_counts = learner.getStateCounts()
            self.env.plot_values(arr_state_counts, ax=ax_C, cmap="Blues", vmin=0, vmax=np.max(arr_state_counts))
            if points_to_add_in_counts_plot is not None:
                self.env.plot_points(points_to_add_in_counts_plot, ax=ax_C, style='x', markersize=5, color="red")
            self._add_count_labels(ax_C, arr_state_counts, fontsize=fontsize_labels, color=color_labels)

            plt.title("[{}FINAL, Learning step {}]\nState visit counts: # visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                      .format(method_name, t_learn+1, state_counts_min, state_counts_mean, state_counts_max))
        else:
            # Add the state counts to the plot of the state value function
            assert self.env.getDimension() == 1, "The environment is expected to be 1D"
            #plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
            axes = fig_V.get_axes()
            assert len(axes) == 2
            axes[1].cla()   # Clear the plot in the axis so that the color of the bars do not get more and more intense which would cover the plot of the state value function
            axes[1].bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.sca(axes[0])  # Go back to the primary axis

        plt.pause(0.1)  # Need to call plt.pause() in order for immediate update of the plot
        plt.draw()

        # Update the policy plot
        if self.fig_policy is not None:
            self._update_policy_plot(self.fig_policy, fontsize=fontsize_labels, color="white", factor_fontsize=0.8, title_prefix=f"[{method_name}FINAL, Learning step {t_learn+1}]\n")
            plt.pause(0.1)
            plt.draw()

    def _update_policy_plot(self, fig, fontsize=14, color="black", factor_fontsize=1.0, title_prefix="", title_suffix=""):
        # Read the environment's shape
        assert len(self.env.getShape()) <= 2, "The environment must be at most 2D"
        shape = self.env.getShape()
        is_env_2d = len(shape) == 2 and shape[0] > 0 and shape[1] > 0

        # Get the current policy
        policy = self.agent.getPolicy()
        colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"

        # Update the policy plot
        if is_env_2d:
            # Create a grid for EACH state having at most 4 actions (going UP, RIGHT, DOWN, LEFT for 2D environments)
            proba_actions_toplot = np.nan * np.ones((3, 3))

            # Create the subplots with the same layout as the states in the environment
            # Note that we use the squeeze=False option so that if the 2D shape is actually a 1D shape (e.g. (1, 5)) then no error is given when referencing axes[1] below.
            axes = fig.subplots(*shape, squeeze=False)
            # Factor to multiply the given fontsize depending on the image (environment) size
            factor_fs = factor_fontsize * np.min((5 / shape[0], 5 / shape[1]))
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    state_1d = np.ravel_multi_index((i, j), shape)
                    for action in range(self.env.getNumActions()):
                        idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                        proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
                    # Clear the axis to avoid having modified colors as new updates are done (because the new image is overlaid on the existing image)
                    axes[i, j].cla()
                    img = axes[i, j].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1)
                    # Remove the axes ticks as they do not convey any information
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                    for action in range(self.env.getNumActions()):
                        idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                        # Recall the first coordinate indexes the row of the matrix representing the actions (vertical axis of the image)
                        # and the second coordinate indexes the column of the matrix representing the actions (horizontal axis of the image)
                        # That's why the text is placed at coordinate (idx_2d[1], idx_2d[0]) instead of (idx_2d[0], idx_2d[1]).
                        # Note that the policy values are shown as percent and using exactly 2 digits (i.e. zero-padded if needed).
                        axes[i, j].text(idx_2d[1], idx_2d[0], "{:02d}".format(int(round(proba_actions_toplot[idx_2d]*100))), color=color, fontsize=fontsize*factor_fs, horizontalalignment="center", verticalalignment="center")
        else:
            # Create a grid for EACH state with the two possible actions (going LEFT or RIGHT)
            # Note that we define this object as an 2D array (as opposed to 1D) because a 2D array is required by the imshow() method used below to generate the image.
            proba_actions_toplot = np.nan * np.ones((1, 2))

            # Create the subplots with the same layout as the states in the environment
            # The 1D environment is laid out horizontally, for better fit of the plot in the figure
            axes = fig.subplots(1, shape[0])
            # Factor to multiply the given fontsize depending on the image (environment) size
            factor_fs = factor_fontsize * 5 / shape[0]
            for i in range(len(axes)):
                state = i
                for action in range(self.env.getNumActions()):
                    proba_actions_toplot[0, action] = policy.getPolicyForAction(action, state)
                # Clear the axis to avoid having modified colors as new updates are done (because the new image is overlaid on the existing image)
                axes[i].cla()
                img = axes[i].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1)
                # Remove the axes ticks as they do not convey any information
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                for action in range(self.env.getNumActions()):
                    # Recall the first coordinate in text() indexes the row of the matrix representing the actions (vertical axis of the image)
                    # and the second coordinate indexes the column of the matrix representing the actions (horizontal axis of the image)
                    # Note that the policy values are shown as percent and using exactly 2 digits (i.e. zero-padded if needed).
                    axes[i].text(0, action, "{:02d}".format(int(round(proba_actions_toplot[0, action]*100))), color=color, fontsize=fontsize*factor_fs, horizontalalignment="center", verticalalignment="center")

        # Finalize plot
        # Activate the current figure so that we can add the title and force drawing (with plt.draw() outside of this method)
        plt.figure(img.get_figure().number)
        plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                                    # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        plt.suptitle(f"{title_prefix}Policy at each state{title_suffix}")

    def _add_count_labels(self, ax, state_counts, fontsize=14, color="orange", factor_fontsize=1.0):
        "Adds the counts in `state_counts` as labels of each cell in the given `ax` axis which is assumed to contain a 2D image"
        # Read the environment's shape and convert it to a 2D array if the environment is 1D in order to facilitate all the processes below that use shape[1]
        # Note that the 1D environment is laid out horizontally (we use e.g. (1, 20) instead of (20, 1))
        assert len(self.env.getShape()) == 2, "The environment must be 2D"
        shape = self.env.getShape()

        # Factor to multiply the given fontsize depending on the image (environment) size
        factor_fs = factor_fontsize * np.min((5 / shape[0], 5 / shape[1]))
        self.env.add_labels(ax, state_counts, fontsize=fontsize*factor_fs, color=color)

    def learn_terminal_state_values(self, learner, t, terminal_state, next_state, reward, info, done_episode=False, envs=None, idx_particle=None, update_phi=False):
        """
        Learns the state and action values for the terminal state in the continuing learning task context

        The action values are set to the same value for all actions, because all actions lead to an environment's start state,
        which is the state chosen after the agent reaches a terminal state, in order to continue with its learning task.

        Arguments:
        learner: Learner
            Learner object responsible for learning the value functions stored in it.
            
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
        # We update just one action for Q and then copy its value to the other Q values
        # In fact, all Q-values for the terminal state are the same because all the actions lead to a start state --defined by env.reset())
        action_anchor = 0

        # Learn
        if t <= 0:
            # We are learning the values of the terminal state at the start of an episode
            # => Do not update the trajectory nor the state count because they were updated at the end of the previous episode.
            # Note that, even if the simulation ends at the given `next_state` we set these flags to False because they concern the update of the trajectory and the counts
            # of the TERMINAL state visited at the previous step, NOT of the `next_state` to which the system transitions (which may happen to be the end state when the simulation
            # ends there).
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
        # TODO: (2024/08/23) If it becomes necessary, we could implement the copy of Q(s,a) for the case of continuous-state environments by performing all possible actions on the same continuous-valued state visited when `done` and learning from those actions (i.e. calling learner.learn())
        # Note that in principle, this copy operation IS necessary because, when learning the action value of the state just BEFORE visiting a terminal state,
        # the average of Q(s,a) over all actions is used in the computation of the TD error (see LeaTD._compute_deltas()).
        if not self.env.isStateContinuous():
            self._copy_action_values_for_terminal_state(learner.getQ(), terminal_state, action_anchor)
            self._copy_action_values_for_terminal_state(learner.getA(), terminal_state, action_anchor)

    def _copy_action_values_for_terminal_state(self, QA, state, action):
        """
        Copies the Q values of a terminal state so that all Q values are the same for all possible actions (as no action is taken at a terminal state!)

        The value of the given state and action are copied to all other actions for the given state stored in the input action values object, Q.

        Arguments:
        QA: ActionValueFunctionApprox
            Object containing the action values or advantage values to modify.

        state: int
            Index of the state whose action values should be copied. Usually a terminal state of the environment.

        action: int
            Index of the action whose value is copied to all other possible actions of the environment.
        """
        for _action in range(self.env.getNumActions()):
            # TODO: (2023/11/23) Generalize this update of the Q-value to ANY function approximation as the following call to _setWeight() assumes that we are in the tabular case!!
            QA._setWeight(state, _action, QA.getValue(state, action))
        # Check that all Q values are the same for the given state
        for _action in range(self.env.getNumActions()):
            assert np.isclose(QA.getValue(state, _action), QA.getValue(state, action)), f"All Q-values are the same for the terminal state {state}:\n{QA.getValues()}"

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
            V, Q, A, n_visits_i, RMSE_by_episode_i, MAPE_by_episode_i, learning_info = \
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
