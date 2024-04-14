# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:04:22 2023

@author: Daniel Mastropietro
@description: Fleming-Viot learners on discrete-time, discrete-state, discrete-action environments.
"""

import numpy as np
import pandas as pd

from Python.lib.agents.learners import AlphaUpdateType, LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambda, LeaTDLambdaAdaptive
from Python.lib.estimators.fv import compute_fv_integral, merge_proba_survival_and_phi, update_phi
from Python.lib.utils import basic, computing

class LeaFV(LeaTDLambda):
    """
    Fleming-Viot learner on a discrete environment defined with the DiscreteEnv class of openAI's gym module based on
    learning episodes.

    The class is designed so that it can be used for both discrete-time and continuous-time Markov processes.

    Learning of value functions (e.g. V(s)) are learned with TD(lambda).

    Arguments:
    env: gym.envs.toy_text.discrete.DiscreteEnv
        The environment where the learning takes place.
        
    N: int
        Number of particles to use for the FV process.

    T: int
        Number of time steps to use for the single Markov chain simulation used to estimate the
        expected reabsorption cycle time, E(T_A), the denominator of the FV estimator of expectations.
        This number is problem dependent and should be large enough to expect a reliable estimation
        of such expectation.

    absorption_set: set
        Set A of absorption states. It should include ALL the states that are NOT of interest
        for the FV exploration because all states that are NOT part of this set are considered
        as valid states for the FV excursion.
        Note that the start state for the single Markov chain simulation used to estimate
        the expected reabsorption cycle time is selected out of this set.

    activation_set: set
        Set of activation states, out of which the start states for the N-particle FV process are selected.

    states_of_interest: (opt) set
        Set of indices representing the states of the environment on which the FV estimation is of interest.
        If `None`, the environment terminal states are defined as states of interest.
        default: None

    probas_stationary_start_state_et: (opt) dict
        Dictionary containing the probability for the selection of the start state to be used to estimate
        the denominator of the FV estimator of expectations, namely the expected reabsorption cycle time, E(T_A).
        See further details in the documentation of the class responsible to simulate Fleming-Viot (e.g. discrete.Simulator).
        default: None

    probas_stationary_start_state_fv: (opt) dict
        Dictionary containing the probability for the selection of the start state to be used for
        the initialization of the N particles of the FV process, which are used to estimate the survival
        probability P(T>t) and the empirical distribution Phi(x,t) that estimates the occupation
        probability of each state x of interest conditional to the process not being absorbed.
        See further details in the documentation of the class responsible to simulate Fleming-Viot (e.g. discrete.Simulator).
        default: None

    criterion: (opt) LearningCriterion
        The learning criterion to use, either AVERAGE or DISCOUNTED.
        Under the AVERAGE learning criterion, episodic learning tasks are converted to a continuing learning task,
        and value functions are defined as *differential* value functions (i.e. where their estimator,
        the return G(t), is corrected by an estimate of the long-run average reward).
        Note that, the value of gamma can still be smaller than 1 in this case (see for instance
        the definition of the state value function proposed by Sutton (2018) on pag. 251/252
        in order to tackle the case of non-ergodic Markov chain induced by a deterministic policies).
        default: LearningCriterion.AVERAGE

    estimate_on_fixed_sample_size: (opt) bool
        Whether to compute the FV estimation of value functions based on groups of particles of fixed size
        determined by their start states and state-actions.
        default: False

    burnin_time: (opt) int or None
        Burn-in time to wait until the empirical mean estimation of Phi(t,x) is considered to have reached
        stationarity, and thus be independent of the initial state and action.
        When `None`, the burn-in time is set to twice the number of particles, 2*N, as the time the FV
        system needs to reach stationarity is proportional to the number of particles in the system.
        default: 0

    TIME_RESOLUTION: (opt) positive float (in capital letters because the concept is that it's a constant)
        Time resolution for the underlying Markov process governing the dynamics of the Fleming-Viot
        particles when they are not absorbed.
        If the underlying Markov process is a discrete-time process, this value should be set to 1.
        Otherwise, it should be set to a small real number compared to the typical inter-event times
        of the continuous-time Markov process, such as 1/lambda for a network system where lambda is the
        event arrival rate.
        default: 1
    """

    def __init__(self, env, N: int, T: int, absorption_set: set, activation_set: set,
                 states_of_interest: set=None,
                 probas_stationary_start_state_et: dict=None,
                 probas_stationary_start_state_fv: dict=None,
                 criterion=LearningCriterion.AVERAGE,
                 estimate_on_fixed_sample_size=False,
                 alpha=0.1, gamma=1.0, lmbda=0.0,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0., func_adjust_alpha=None,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 burnin_time=0, TIME_RESOLUTION=1,
                 debug=False):
        super().__init__(env, criterion=criterion, task=LearningTask.CONTINUING, alpha=alpha,  gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha,
                         alpha_update_type=alpha_update_type, adjust_alpha_by_episode=adjust_alpha_by_episode,
                         alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         store_history_over_all_episodes=True,  # We set this to True because FV is a CONTINUING learning task
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed, debug=debug)

        #-- Fleming-Viot simulation parameters
        self.N = N          # Number of particles in the FV simulation used to estimate the QSD Phi(t,x) for every state x at discrete time t
        # Number of particles per start state and start state-action, used to update the respective Phi(t,x; s) and Phi(t,x; s,a)
        self.N_for_start_state = [0]*self.env.getNumStates()
        self.N_for_start_action = [0]*self.env.getNumActions()
        self.N_for_start_state_action = [None]*self.env.getNumStates()
        for s in range(len(self.N_for_start_state_action)):
            self.N_for_start_state_action[s] = [0]*self.env.getNumActions()
        # Maximum number of time steps that are allowed to be run in each episode of the single Markov chain that estimates the expected reabsorption time E(T_A)
        self.T = T
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set ({}).".format(type(absorption_set)))
        if not isinstance(activation_set, set):
            raise ValueError("Parameter `activation_set` must be a set ({}).".format(type(activation_set)))
        self.absorption_set = absorption_set
        self.activation_set = activation_set
        # The complement of the absorption set A
        self.active_set = set(self.env.getAllValidStates()).difference(self.absorption_set)
        self.states_of_interest = self.env.getTerminalStates() if states_of_interest is None else states_of_interest
        # Dictionary that stores the optional stationary probability of the start states of the single chain simulation
        # and of the N-particle Fleming-Viot process.
        # If None, uniformly random distributions are used.
        self.probas_stationary_start_state_et = probas_stationary_start_state_et
        self.probas_stationary_start_state_fv = probas_stationary_start_state_fv

        # Note: The average reward that is estimated by the FV learner is already stored in the GenericLearner class
        # which is the superclass of the Learner class which in turn is the superclass of the LeaTDLambda class
        # from which this LeaFV class inherits.

        # Time resolution for the underlying Markov process on which Fleming-Viot acts
        # If the process is discrete-time, the time resolution should be set to 1.
        self.TIME_RESOLUTION = TIME_RESOLUTION

        #-- Attributes that are defined in reset()
        # List of absorption times
        # Note that these are NOT the survival times but the (absolute) absorption times. See for instance the comment about parameter `t` in learn_at_absorption()
        # where we explicitly state that `t` is the system's clock and it is NOT used to compute the survival time.
        self.absorption_times = None
        self.start_states = None
        self.start_actions = None
        self.start_state_actions = None
        # Dictionary indicating the indices in self.dict_phi (for each state of interest x)
        # that inform the last Phi(x) row index whose time 't' is BEHIND the respective absorption time stored in self.absorption_times.
        # Note that, for each x, dict_last_indices_phi_prior_to_absorption_times[x] contains as many indices as the length of self.absorption_times).
        # This dictionary allows us to quickly retrieve, for each x, the first index in Phi(x) that should contribute
        # to the computation of the Phi(x) sum for all the times contained between the previous absorption time
        # and the absorption time currently being processed.
        # Ex:
        # self.absorption_times = [0, 3, 8, 13]     (integer-valued!)
        # self.dict_phi[x=9] = {'t': [0, 2, 4, 5, 7, 12], 'Phi': [0, 0.1, 0.0, 0.1, 0.2, 0.2]}
        # self.dict_last_indices_phi_prior_to_absorption_times[x=9] = [0, 1, 4, 5]
        self.dict_last_indices_phi_prior_to_absorption_times = None
        # Phi estimators and their cumulative sum
        # (this is used to update the FV integral when a new absorption is observed,
        # as long as the absorption times are observed in order and we derive just ONE single SURVIVAL time from each
        # of the observed absorption times --i.e. this is NOT suitable when updating the FV integral that contributes
        # to the estimation of value functions V(x) and Q(x,a) as these estimates require the use of survival times
        # that DEPEND on the state x and action a!)
        self.dict_phi = None
        self.dict_phi_sum = None
        # Survival times and empirical mean estimates of Phi(t,x) for each start state and each start state-action, i.e. measured from each time the state (or state-action) is visited
        # during a trajectory just before an absorption (leading to a survival time) happens.
        self.dict_survival_times_for_state = None
        self.dict_survival_times_for_state_action = None
        self.dict_phi_for_state = None
        self.dict_phi_for_state_action = None
        # FV integral value for each state of interest x (updated every time a new absorption is observed)
        self.dict_integral = None
        self.dict_integral_for_state = None
        self.dict_integral_for_state_action = None

        # Expected value of the discounted state value at absorption, which is part of the decomposition of the value functions in FV-estimated part and rest in e.g.
        # Q(x,a) = Q_FV(x,a) + Q_after_absorption(x,a) = Q_FV(x,a) + E[ gamma^t_abs * V(X_abs) ]
        self.expected_discounted_state_value_at_absorption = 0.0
        self.n_expected_discounted_state_value_at_absorption = 0    # Sample size associated to the estimated expected discounted state value at absorption

        # Whether to use fixed group of particles to estimate the value functions (based on the particle's start state and start state-actions)
        self.estimate_on_fixed_sample_size = estimate_on_fixed_sample_size

        # Burn-in time that should be waited to consider that Phi(t,x) has achieved stationarity,
        # and thus independent of the starting state and action where the particles contributing to its estimation
        # can be considered to have reached stationary behaviour.
        # In fact, when computing the integral of P(T>t; s,a) * Phi(t,x) for each state of interest x,
        # we are assuming stationarity when writing Phi(t,x) instead of Phi(t,x; s,a), i.e. Phi(t,x) is assumed to NOT depend on
        # the "start" state-actions (s,a)!
        self.burnin_time = self.N*2 if burnin_time is None else burnin_time

        self.reset()

    def reset(self, reset_episode=False, reset_value_functions=False):
        super().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)

        #-- Reset the (integer) absorption times
        # We initialize the list of absorption times to 0 to make the update of the Phi contribution to the FV integral
        # easier to carry out, with less IF conditions that check whether a particle has already been absorbed.
        # See also comment about this being the absorption times, as opposed to the survival times, in the constructor, where this attribute is defined.
        self.absorption_times = [0]

        #-- Reset attributes related to the iterative update of the FV integral when survival times are observed in order
        self.dict_last_indices_phi_prior_to_absorption_times = dict()
        self.dict_phi = dict()
        self.dict_phi_sum = dict()
        self.dict_integral = dict()
        for x in self.states_of_interest:
            self.dict_last_indices_phi_prior_to_absorption_times[x] = [0]
            # Note that time is allowed to be float (because the class should still be valid to deal with continuous-time Markov processes)
            self.dict_phi[x] = pd.DataFrame([[0.0, 0.0]], columns=['t', 'Phi'])
            self.dict_phi_sum[x] = pd.DataFrame([[0.0, 0.0]], columns=['t', 'Phi'])
            self.dict_integral[x] = 0.0

        #-- Reset attributes related to the estimation of the FV integral for each start state and start state-action
        self.start_states = [-1]*self.N                 # -1 means "missing"
        self.start_actions = [-1]*self.N                # -1 means "missing"
        self.start_state_actions = [(-1, -1)]*self.N    # Note: I checked that modifying one element of this list does NOT modify the other elements!

        # Survival probability functions are defined for each state in the set of active states (i.e. all the states that are NOT in the absorption set A)
        self.dict_survival_times_for_state = dict()
        self.dict_survival_times_for_state_action = dict()
        self.dict_phi_for_state = dict()
        self.dict_phi_for_state_action = dict()
        self.dict_integral_for_state = dict()
        self.dict_integral_for_state_action = dict()
        for s in self.active_set:
            # Survival functions and empirical mean estimate of Phi(t,x) as functions of each state and of each state-action
            self.dict_survival_times_for_state[s] = [0.0]
            self.dict_survival_times_for_state_action[s] = dict()
            self.dict_phi_for_state[s] = dict()
            self.dict_phi_for_state_action[s] = dict()
            self.dict_integral_for_state[s] = dict()
            self.dict_integral_for_state_action[s] = dict()
            for x in self.states_of_interest:
                self.dict_phi_for_state[s][x] = pd.DataFrame([[0, float(s in self.env.getTerminalStates())]], columns=['t', 'Phi'])
                # TODO: (2024/01/27) Case before the estimate_on_fixed_sample_size mode... Do we need to use this when self.estimate_on_fixed_sample_size = False?
                #self.dict_phi_for_state[s][x] = pd.DataFrame(columns=['t', 'Phi'])
                self.dict_integral_for_state[s][x] = 0.0
            for a in range(self.env.getNumActions()):
                self.dict_survival_times_for_state_action[s][a] = [0.0]
                self.dict_phi_for_state_action[s][a] = dict()
                self.dict_integral_for_state_action[s][a] = dict()
                for x in self.states_of_interest:
                    self.dict_phi_for_state_action[s][a][x] = pd.DataFrame([[0, float(s in self.env.getTerminalStates())]], columns=['t', 'Phi'])
                    # TODO: (2024/01/27) Case before the estimate_on_fixed_sample_size mode... Do we need to use this when self.estimate_on_fixed_sample_size = False?
                    #self.dict_phi_for_state_action[s][a][x] = pd.DataFrame(columns=['t', 'Phi'])
                    self.dict_integral_for_state_action[s][a][x] = 0.0

    def deprecated_store_start_states(self, envs):
        "Stores the start states of the particles as stored in the CURRENT state of the input environments"
        # Compute the distribution of particles across the different states
        self.start_states = [env.getState() for env in envs]
        unique_start_states, count_start_states = np.unique(self.start_states, return_counts=True)
        dist_start_states = count_start_states / self.N
        for s, count in zip(unique_start_states, count_start_states):
            self.N_for_start_state[s] = count
        import pandas as pd
        print(f"Distribution of particles at start of simulation:\n{pd.DataFrame(np.c_[unique_start_states, count_start_states, dist_start_states], columns=['state', 'freq', 'dist'])}")

    def learn(self, t, state, action, next_state, reward, done, info, envs=None, idx_particle=None, update_phi=False):
        """
        Learns the value functions using TD learning and updates the particle's trajectory in the environment representing the particle

        Arguments:
        envs: (opt) list of Environment
            All the environments associated to the FV particles.

        idx_particle: (opt) int
            Index of `envs` representing the particle whose trajectory history should be updated with the new transition.
            This is the particle that has experienced this transition and it is needed to be able to find all the states and actions
            that have been visited by the particle at the absorption event, which are used to estimate the survival probability given
            each visited state and action.
            default: None

        update_phi: (opt) bool
            Whether the estimate of the empirical mean of conditional state occupation, Phi(t,x) for every state of interest x
            should be updated with this particle transition.
            This should be True ONLY when `env` is the environment associated to an FV particle, and NOT for instance when it is
            the environment associated to a particle that is allowed to explore the whole environment freely.
            default: False
        """
        if info.get('learn_from_superclass', True):
            # Learn the value function using the superclass learner
            super().learn(t, state, action, next_state, reward, done, info)

        if envs is not None:
            assert envs[idx_particle].getStoreTrajectoryFlag(), "The environment associated to the FV particle on which the value functions learning is carried out must have stored the observed trajectory"

            if update_phi:
                # Update the empirical distribution of both state and next_state when any of them is part of the set of states of interest.
                # This should ONLY happen when the particle (env) transitioning is an FV particle, as opposed to e.g. a particle that can freely explore the whole environment
                # (because in that case, Phi has no role to play).
                # Note that `t` is the ABSOLUTE time, i.e. the time elapsed since the start of the FV simulation which is
                # independent of any survival time, which on the contrary is relative to the start position (and action) of the particle
                # that is eventually absorbed (i.e. the survival time is measured w.r.t. the time at which the particle was
                # at each state s visited by the particle's trajectory and taking action `a`).
                # Note however that at the moment of computing the FV integral, there will be an alignment between the
                # survival time associated to the survival probability of a particle starting at state s and action `a`
                # and the time t stored in the Phi(t,x) function considered on the state of interest x.
                if self.estimate_on_fixed_sample_size:
                    assert self.getStartAction(idx_particle) >= 0, f"The start action must be a valid start action, i.e. there must be at least one particle that started at that action (s={self.getStartState(idx_particle)}, a={self.getStartAction(idx_particle)})"
                    self._update_phi_function_fixed_sample_size(self.getStartState(idx_particle), self.getStartAction(idx_particle), t, state, next_state)
                    #print(f"[learn] Updated Phi(t,x; s={self.start_states[idx_particle]}):\n{self.dict_phi_for_state[self.start_states[idx_particle]]}")
                else:
                    self.update_phi(t, state, next_state)

            # Store the trajectory in the particle (environment) that has evolved
            # (so that we are able to retrieve its history when learning the value functions by the FV estimator)
            self._update_particle_trajectory(envs[idx_particle], t, state, action, next_state, reward)
            #print(f"[debug-learn] Trajectory of particle {idx_particle} UPDATED:\n{envs[idx_particle].getTrajectory()}")

    def learn_at_absorption(self, envs, idx_particle, t, state_absorption, next_state):
        """
        Learns the value functions at the time of the particle's absorption for EVERY state and action in the trajectory of
        the particle prior to absorption, using:
        - the FV estimator of the value function of the killed process
        - the n-step bootstrap estimator of the value function, where n is the number of steps taken
        until absorption as measured from the time the particle was at the state and action whose value
        is now updated).

        The method also updates Phi(t,x) for every state of interest x as a result of reactivation
        from the absorption state `state_absorption` to `next_state`.

        Arguments:
        envs: list of Environment (particle)
            Environments associated to all the FV particles in the system.
            These are used to:
            (i) extract the trajectory of the absorbed particle before absorption.
            (ii) update the empirical mean phi(t,x) as a function of every state and state-action visited by all particles
            prior to the current absorption event.

        idx_particle: int
            Index of `envs` representing the absorbed particle.
            This is used to extract the states and actions visited by the particle prior to absorption
            alongside the times they were visited at.

        t: positive float
            Simulation time elapsed since its start, which is updated whenever ANY particle moves.
            It is the time at which the absorption happened, i.e. the time at which the Markov process moved to the absorbed state.
            Hence t coincides with the absorption time.
            This value is used to update Phi(t,x) and to update the self.absorption_times list that keeps track of
            all the SYSTEM absorption times.
            Note that it is NOT used to compute the survival time of the particle that has been absorbed,
            since the survival time is computed from the particle's trajectory stored SPECIFICALLY in the environment
            associated to the particle (whose row index has the actual clock of the particle's movement).

        state_absorption: int
            Index representing the absorption state, which is used for the bootstrap estimation part.

        next_state: int
            Index representing the state where the absorbed particle was reactivated to, which is used
            for the update of Phi(t,x) for each state of interest x.
        """
        #-- Update Phi based on the new reactivated position of the particle after the absorption
        # This is important because Phi could have already experienced an update, which is the case when the absorption occurred from a state of interest x
        # as this implies that the occupation frequency of x changed when the particle was absorbed.
        self.update_phi(t, state_absorption, next_state)

        # Update the Phi(t,x) as a function of the start state, for all states in the active set
        self.update_phi_function(envs)

        # Update the list of observed absorption times
        # (this is used when updating the FV integral, normally by update_integral(), where the absorption time is assumed to be a survival time)
        self.absorption_times += [t]

        #-- Learn the value function using the FV estimator
        # This means: update the value functions for all states and actions visited by the particle before absorption

        # Compute the number of steps it took the absorbed particle to become absorbed (t_particle_abs)
        # Note that, in principle, we could use either this t_particle_abs as absorption time of the particle OR the FV system's time plus 1 `(t + 1)`,
        # (i.e. either the particle's clock OR the FV system's clock) to compute the survival time for each start state s.
        # Note that the scaling of the FV integral is different depending on which of the two approaches we use for the absorption time:
        # - when using the particle's clock, the scaling is 1 because we are considering each particle separately and its change rate is 1 change per unit time.
        # - when using the FV system's clock, the scaling is 1/N because the change rate of the system is N per unit time, which defines the uniformization
        # constant that discretizes the originally continuous-time FV process.
        # Note that the particle's clock is stored as the INDEX of the data frame storing the particle's trajectory.
        df_trajectory_absorbed_particle = envs[idx_particle].getTrajectory()
        # Absorption time in the particle's clock: it's the maximum index value of the data frame storing the particle's trajectory
        t_particle_abs = df_trajectory_absorbed_particle.index[-1]
        assert t_particle_abs > 0, f"The absorption time of the particle is > 0 ({t_particle_abs})"
        print(f"[debug-learn_at_absorption] Trajectory of the absorbed particle #{idx_particle} at system's clock time={t}, and internal clock time={t_particle_abs}:\n{df_trajectory_absorbed_particle}\n")
        # TODO: (2024/01/17) IMPORTANT: Collect ALL survival times observed since visiting each of the UNIQUE states and state-actions in the absorbed particle's past trajectory
        # Goal: update P(T>t; s) and P(T>t; s,a) only ONCE for each state and each state-action.
        for row in df_trajectory_absorbed_particle.iterrows():
            # Internal time of the absorbed particle (it ALWAYS takes the values 0, 1, 2, ...)
            # (which is NOT the simulation time because the simulation time increases by 1 at the movement of ANY particle)
            t_particle = row[0]
            row_data = row[1]

            # Skip the first record because the trajectory stores the initial state as if the first action would be to "place the particle at its initial state"
            # This implies that such initial state record has action = -1, as there is no environment action to "place" a particle at a state.
            # Thus, if we don't skip the first record, we would get an error about invalid action when processing the calculation of the FV integral below.
            if t_particle == 0:
               continue

            # Compute the survival time (i.e. until killing) using as origin the time when the particle visited the state and action
            # currently retrieved from the particle's trajectory. Note that this visiting time is `t_particle - 1` and NOT `t_particle`
            # because each index of the trajectory data frame (which is precisely t_particle) gives the time at which the particle visited `next_state`
            # (NOT the time at which the particle visited `state`) => therefore, the time the particle visited `state` and took action `action` is `t_particle - 1`.
            # If we don't do this, we would get t = 0 duplicated when estimating the survival time probability distribution P(T>t; state, action)
            # from the currently analyzed trajectory, and that would yield the usual error I get when such repeated value happens in P(T>t), namely e.g.:
            #   "AssertionError: The length of the first pair of merged lists is the same as that of the second pair of merged lists (3, 3, 2, 2)"
            # triggered by the basic.merge_values_in_time() called when merging the survival probability function and Phi(t,x).
            t_surv = t_particle_abs - (t_particle - 1)  # Use this if using the particle's clock to compute the survival time --> in this case the value of uniform_jump_rate in _compute_fv_deltas() should be 1
            #t_surv = (t + 1) - row_data['time']        # Use this if using the system's clock to compute the survival time --> in this case the value of uniform_jump_rate in _compute_fv_deltas() should be self.N (the number of particles)
            state = int(row_data['state'])
            action = int(row_data['action'])
            assert state in self.dict_survival_times_for_state.keys(), f"All states visited by the particle prior to absorption need to be in the active set of states (s={state})"
            assert action >= 0, f"The action must be >= 0: action={action}"

            delta_V, delta_Q, state_value_killed_process, action_value_killed_process = self._compute_fv_deltas(t_surv, state_absorption, state, action)

            #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_over_all_episodes[state], self.getAlphaForState(state)))
            # Store the learning rates to be used in the value functions update
            self.store_learning_rate(self.getAlphasByState())
            self._updateZ(state, action, self.lmbda)
            self._updateV(delta_V)
            self._updateQ(delta_Q)

            # Update alpha for the next iteration for the "by state counts" update
            # NOTE that we only update alpha when the value V(state) was actually updated (i.e. delta(V) != 0),
            # o.w. we don't want to decrease alpha when no signal was received from the rare rewards outside A.
            #print("Learn: state = {}, next_state = {}, done = {}".format(state, next_state, done))
            if delta_V != 0.0:
                self._update_state_counts(t, state)
                self._update_alphas(state)
            # TODO: (2024/01/03) ALSO update the alphas by state-action (to be defined in the GenericLearner class first)

            if delta_Q != 0.0 and state in self.env.getTerminalStates():
                # TODO: (2024/01/17) Move this piece of code to a function as this is done already at two places at least
                # Copy the Q value just learned to all the other actions in a terminal state, because no action is taken at the terminal state, so all Q values should have the same value for all the actions
                for _action in range(self.env.getNumActions()):
                    # TODO: (2023/11/23) Generalize this update of the Q-value to ANY function approximation as the following call to _setWeight() assumes that we are in the tabular case!!
                    self.getQ()._setWeight(state, _action, self.getQ().getValue(state, action))
                # Check that all Q values are the same for the terminal state
                for _action in range(self.env.getNumActions()):
                    assert np.isclose(self.getQ().getValue(state, _action), self.getQ().getValue(state, action)), \
                        f"All Q-values are the same for the terminal state {state}:\n{self.getQ().getValues()}"

        # Reset the trajectory of the absorbed particle for the next absorption event to the next state which is the reactivated state
        envs[idx_particle].reset_trajectory(next_state)

    def learn_at_absorption_fixed_sample_size(self, envs, idx_particle, t_surv, t_phi, state_absorption, next_state):
        """
        Learns the value functions at the time of the particle's absorption for the state and action at which the particle STARTED,
        an information that is retrieved by the methods self.getStartState() and self.getStartAction().

        The method also updates Phi(t,x) for the given start state and action, for every state of interest x as a result of reactivation
        from the absorption state `state_absorption` to `next_state`.

        Arguments:
        t_surv: positive float
            Survival time as measured in the internal clock of the absorbed particle, used in the estimation of the survival probability P(T>t)
            associated to the start state of the absorbed particle.

        t_phi: positive float
            Time used in the estimation of the conditional occupation probability Phi(t,x), which should be the SYSTEM time of the group of FV particles
            to which the absorbed particle belongs.
        """
        assert t_surv <= t_phi, f"The particle's clock ({t_surv}) must be at MOST equal to the system's clock ({t_phi})"
        #-- Update Phi based on the new position of the particle following reactivation after the absorption
        # This is important because Phi could have already experienced an update, which is the case when the absorption occurred from a state of interest x
        # as this implies that the occupation frequency of x changed when the particle was absorbed.
        start_state = self.getStartState(idx_particle)
        start_action = self.getStartAction(idx_particle)
        self._update_phi_function_fixed_sample_size(start_state, start_action, t_phi, state_absorption, next_state)
        # Check if the Phi function was REALLY updated by the above call (i.e. if a new entry with time `t_phi` was added to Phi --which doesn't happen when the value of Phi
        # at x doesn't change due to the reactivation of the particle from state `state_absorption` to `next_state` just because no state tracked by Phi is either one of those states).
        # If Phi was NOT updated by the above call, we add `t_phi` to the data frame of Phi(t,x) values below because we need it for:
        # - updating the Phi contribution to the FV integral at the CURRENT absorption time (as this absorption time defines the end time on which the Phi values are summed)
        # - updating the Phi contribution to the FV integral at the NEXT absorption time (as this absorption time defines the start time on which the Phi values will be summed)
        for x in self.states_of_interest:
            if self.dict_phi_for_state[start_state][x]['t'].iloc[-1] < t_phi:
                _phi_last = self.dict_phi_for_state[start_state][x]['Phi'].iloc[-1]
                self.dict_phi_for_state[start_state][x] = pd.concat([self.dict_phi_for_state[start_state][x],
                                                                     pd.DataFrame([[t_phi, _phi_last]], index=[self.dict_phi_for_state[start_state][x].shape[0]], columns=['t', 'Phi'])],
                                                                    axis=0)
            if self.dict_phi_for_state_action[start_state][start_action][x]['t'].iloc[-1] < t_phi:
                _phi_last = self.dict_phi_for_state_action[start_state][start_action][x]['Phi'].iloc[-1]
                self.dict_phi_for_state_action[start_state][start_action][x] = pd.concat([self.dict_phi_for_state_action[start_state][start_action][x],
                                                                                         pd.DataFrame([[t_phi, _phi_last]], index=[self.dict_phi_for_state_action[start_state][start_action][x].shape[0]], columns=['t', 'Phi'])],
                                                                                        axis=0)

        #-- Learn the value function using the FV estimator on the group of particles that started at the same state at which the absorbed particle started
        delta_V, delta_Q, state_value_killed_process, action_value_killed_process = self._compute_fv_deltas_fixed_sample_size(t_surv, state_absorption, start_state, start_action)

        #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_over_all_episodes[start_state], self.getAlphaForState(start_state)))
        # Store the learning rates to be used in the value functions update
        self.store_learning_rate(self.getAlphasByState())
        self._updateZ(start_state, start_action, self.lmbda)
        self._updateV(delta_V)
        self._updateQ(delta_Q)

        # Update alpha for the next iteration for the "by state counts" update
        # NOTE that we only update alpha when the value V(state) was actually updated (i.e. delta(V) != 0),
        # o.w. we don't want to decrease alpha when no signal was received from the rare rewards outside A.
        #print("Learn: state = {}, next_state = {}, done = {}".format(state, next_state, done))
        if delta_V != 0.0:
            self._update_state_counts(t_phi, start_state)
            self._update_alphas(start_state)
        # TODO: (2024/01/03) ALSO update the alphas by state-action (to be defined in the GenericLearner class first)

        if delta_Q != 0.0 and start_state in self.env.getTerminalStates():
            # TODO: (2024/01/17) Move this piece of code to a function as this is done already at two places at least
            # Copy the Q value just learned to all the other actions in a terminal state, because no action is taken at the terminal state, so all Q values should have the same value for all the actions
            for _action in range(self.env.getNumActions()):
                # TODO: (2023/11/23) Generalize this update of the Q-value to ANY function approximation as the following call to _setWeight() assumes that we are in the tabular case!!
                self.getQ()._setWeight(start_state, _action, self.getQ().getValue(start_state, start_action))
            # Check that all Q values are the same for the terminal state
            for _action in range(self.env.getNumActions()):
                assert np.isclose(self.getQ().getValue(start_state, _action), self.getQ().getValue(start_state, start_action)), \
                    f"All Q-values are the same for the terminal state {start_state}:\n{self.getQ().getValues()}"

    def update_integral(self, t_absorption, fixed_sample_size=True):
        """
        Updates the FV integral based on the given absorption time

        This update assumes that all absorption times are equal to SURVIVAL times contributing to the estimator of P(T>t)
        which are assumed to occur in increasing order.

        The following attributes are updated by this call:
        - self.absorption_times is updated with the given `t_absorption` time, which is added at the end of the list of observed absorption times.
        Note that, for the purpose of the FV integral update done here, absorption times are assumed to be survival times, as they are assumed to be
        the FIRST absorption time observed for each particle, which clearly coincides with their first observed survival time (i.e. only survival times
        coming from the first absorption event of each particle are considered for contribution to the estimation of P(T>t)).
        - self.dict_phi_sum is updated with the new cumulative sum of the Phi values up to time `t_absorption` at each state of interest x stored in self.states_of_interest,
        - self.dict_integral is updated with the new FV integral value of Integral{ P(T>t)*Phi(t,x) } after observing `t_absorption` as the latest and
        so far largest survival time contributing to the estimation of P(T>t). This is done for each state of interest x stored in self.states_of_interest.

        t_absorption: positive float
            Latest observed absorption time, which is assumed to be the latest observed survival time when estimating the survival probability P(T>t).
            These absorption/survival times are assumed to be observed in order.
            Both these conditions are satisfied when the only times contributing to the estimation of the survival probability P(T>t) are the FIRST
            absorption times of each particle in the FV system. Normally, this is implemented in discrete.Simulator._run_simulation_fv().

        fixed_sample_size: (opt) bool
            Whether the integral should be updated based on self.N survival times contributing to the estimation of the survival probability P(T>t).
            This is the case, for instance, when we only consider the FIRST survival time of each particle as contribution to the estimation of P(T>t).
            When False, the estimate of P(T>t) is given by (1 - k/n) for k = 1, ..., n at survival times t(1) < t(2) < ... < t(n) = t_absorption
            observed so far.
            default: True
        """
        assert t_absorption > 0, f"The absorption time must be positive ({t_absorption})"
        self._update_absorption_times_and_phi(t_absorption)
        self._update_phi_contribution(t_absorption)
        self._update_integral(fixed_sample_size=fixed_sample_size)

    def update_phi(self, t, state, next_state):
        "Updates the empirical mean phi(t, x) for x in intersection({state, next_state}, states_of_interest)"
        self.dict_phi = update_phi(self.env, self.N, t, self.dict_phi, state, next_state)

    def update_phi_function(self, envs):
        """
        Updates the empirical mean phi(t, x) for all states x listed in the self.states_of_interest attribute for all the historical discrete times t
        of which we have information at the time of estimation (e.g. at the absorption time of a particle).

        The value of phi(t, x) is considered a function of every start state outside the absorption set A (when phi(t,x) is used to estimate the state value function)
        or a function of every start state-action of states outside A (when phi(t,x) is used to estimate the action value function).

        The update is done for all the states and actions that are visited by all the particles passed in the input list `envs`
        whose values (of the states and actions to update) are taken from the particle trajectories stored in each environment present in the list.

        In particular, the above sentence implies that we can run update_phi_function() at the time we need to compute the phi(t,x; s,a) estimates,
        normally at the time of a new absorption of a particle in the FV system.

        Note that the times at which phi(t, x) is estimated for each state and for each state-action are CONSECUTIVE integers, because Phi is estimated
        for EACH time step of a particle, which are integer-valued in a discrete-time setting.

        Arguments:
        envs: list of Environments
            List containing the environments associated to the particles whose stored trajectories are used to compute the empirical mean phi(t, x).
        """
        #--------------------- Auxiliary functions ---------------------------
        def update_phi_internal(dict_phi, t, x, empirical_mean, last_time_inserted_in_phi: float):
            """
            Updates Phi(t,x) using the empirical_mean as its estimate

            IMPORTANT: For this function to work properly, it is required that the values of column 't' in the Phi(t,x) data frame
            coincide with its row indices (as the value of Phi(t,x) is retrieved by using `.loc[t]` when updating an existing entry).
            This requirement is assured via an assertion in this function that checks that every new record inserted has its index
            equal to its 't' value.

            Parameter last_time_inserted_in_phi contains the last time value that has been inserted for Phi(t,x)
            which is used to know whether the estimate should be added to the Phi(t,x) data frame
            or should replace an already existing record.

            Arguments:
            dict_phi: dict
                Dictionary indexed by the states of interest of which `x` should be part, which is updated by this function.

            t: int
                Time step to which the empirical mean estimate corresponds.

            x: int
                State of interest whose Phi(t,x) is estimated with the empirical mean value.
                It should be a key in the dict_phi dictionary.

            empirical mean: non-negative float
                Empirical mean to store in in the data frame referred by `dict_phi[x]`.

            last_time_inserted_in_phi: float (although normally is int, as the integer value passed in `t` is compared with this value)
                The largest time that was inserted in dict_phi[x] so far.
                This is used to know whether the empirical mean value should be added to the end of the data frame referred by `dict_phi[x]`
                or should replace an existing record in `dict_phi[x]`.

            Return: float
            The updated value for last_time_inserted_in_phi, after the update of Phi(t,x) (stored in the input argument dict_phi) took place.
            """
            if t <= last_time_inserted_in_phi:
                # We are updating Phi(t,x) for a time already stored in dict_phi
                dict_phi[x].loc[t] = [t, empirical_mean]
            else:
                # We are estimating Phi(t,x) at a new time not yet stored in dict_phi
                dict_phi[x] = pd.concat([dict_phi[x],
                                        pd.DataFrame([[t, empirical_mean]], columns=self.dict_phi_for_state[s][x].columns, index=[len(dict_phi[x])])],
                                        axis=0)
                assert dict_phi[x].index[-1] == t, f"The index of the last record just inserted in the data frame storing Phi(t,x) ({dict_phi[x].index[-1]}) must coincide with the value of 't' of the inserted record (t={t})"
                last_time_inserted_in_phi = t

            return last_time_inserted_in_phi
        #--------------------- Auxiliary functions ---------------------------

        actions = np.arange(self.env.getNumActions())

        # Iterate on all states for which Phi(t,x) would be computed when the trajectory started at that state
        # The start state of such trajectories are the states in the active set of states, i.e. the complement of the absorption set A.
        last_tt_inserted_in_phi = dict()
        last_tt_inserted_in_phi_actions = dict()
        for s in self.active_set:
            last_tt_inserted_in_phi[s] = dict()
            last_tt_inserted_in_phi_actions[s] = dict()
            for a in actions:
                last_tt_inserted_in_phi_actions[s][a] = dict()
            for x in self.states_of_interest:
                last_tt_inserted_in_phi[s][x] = self.dict_phi_for_state[s][x]['t'].iloc[-1] if len(self.dict_phi_for_state[s][x]) > 0 else -1
                for a in actions:
                    last_tt_inserted_in_phi_actions[s][a][x] = self.dict_phi_for_state_action[s][a][x]['t'].iloc[-1] if len(self.dict_phi_for_state_action[s][a][x]) > 0 else -1
            # List of indices in each particle's trajectory where the particle was positioned at state `s`, if any
            # This is used to be able to measure the time since each particle visited state s, which would eventually update the value of phi(t,x) for each state of interest x
            indices_trajectory_at_state_s = [np.array([], dtype=int)]*self.N
            particles_that_visited_s = []
            indices_trajectory_at_state_s_actions = [None]*len(actions)
            particles_that_visited_s_actions = [None]*len(actions)
            # Initialize each element of the above lists as an empty list or array with DIFFERENT memory address
            # (which is NOT the case if we simply initialize the list as `[[]]*len(actions)`)
            for a in actions:
                indices_trajectory_at_state_s_actions[a] = [np.array([], dtype=int)] * self.N
                particles_that_visited_s_actions[a] = []
            for p in range(self.N):
                df_trajectory = envs[p].getTrajectory()
                # Find the indices in the particle's trajectory where the particle was at state s
                # Because the state of a particle at each time step corresponds to the state where it moved AFTER the action was taken at the state given in column 'state'
                # of df_trajectory, we use the 'next_state' column (NOT 'state') to retrieve the "current" state of the particle.
                indices_trajectory_at_state_s[p] = np.where(df_trajectory['next_state'] == s)[0]  # Because of the output of np.where() we need to retrieve the 0-th element
                particles_that_visited_s += [p] if len(indices_trajectory_at_state_s[p]) > 0 else []
                for a in actions:
                    # Note that we look for `s` in column 'state' (and NOT 'next_state' as done before)
                    # because we are interested in finding the state where the particle was BEFORE taking the action.
                    # Also recall that the very first record in the particle's trajectory has action = NaN (see EnvironmentDiscrete.reset_trajectory() method).
                    indices_trajectory_at_state_s_actions[a][p] = np.where((df_trajectory['state'] == s) & (df_trajectory['action'] == a))[0]
                    particles_that_visited_s_actions[a] += [p] if len(indices_trajectory_at_state_s_actions[a][p]) > 0 else []

            # Iteration on the times tt at which Phi(tt,x) can be updated, which are assumed to be discrete
            # The iteration will go on insofar as there are particles whose position at time tt after visiting state s are available
            tt = -1
            done_trajectories = False
            while not done_trajectories:
                tt += 1     # We assume a discrete time process!
                n_particles_tt_after_state_s = 0  # Number of particles whose position can be measured at time tt after visiting state s
                if tt == 0:
                    # In this case, the empirical mean is either 0 or 1, i.e. it is simply whether the analyzed state s is among the states of interest
                    # as we are looking at the proportion of particles starting at s that are in x at time 0, i.e. at the time where they are in s!
                    for x in self.states_of_interest:
                        empirical_mean = float( s == x )
                        last_tt_inserted_in_phi[s][x] = update_phi_internal(self.dict_phi_for_state[s], tt, x, empirical_mean, last_tt_inserted_in_phi[s][x])
                        # Initialize the Phi(t,x) for each possible start state-action to the empirical mean of being at x
                        for a in actions:
                            last_tt_inserted_in_phi_actions[s][a][x] = update_phi_internal(self.dict_phi_for_state_action[s][a], tt, x, empirical_mean, last_tt_inserted_in_phi_actions[s][a][x])
                else:
                    n_particles_tt_after_state_s_actions = [0]*len(actions)
                    empirical_mean = 0.0
                    empirical_mean_actions = [0.0]*len(actions)
                    # Go over each particle that visited s in its history and check where they were at time tt of their RESPECTIVE clock
                    for p in particles_that_visited_s:
                        df_trajectory = envs[p].getTrajectory()

                        # 1) Update the empirical mean for the given start state s
                        for idx in indices_trajectory_at_state_s[p]:
                            try:    # This `try` is for the retrieval of the `idx + tt` row from the trajectory which may not exist
                                position_of_particle_at_time_tt_after_state_s = df_trajectory['next_state'].iloc[idx + tt]
                                for x in self.states_of_interest:
                                    empirical_mean += int(position_of_particle_at_time_tt_after_state_s == x)
                                n_particles_tt_after_state_s += 1
                            except IndexError:
                                pass

                        # 2) Update the empirical mean for each possible start action at the given start state s
                        for a in actions:
                            for idx in indices_trajectory_at_state_s_actions[a][p]:
                                try:  # This `try` is for the retrieval of the `idx + tt` row from the trajectory which may not exist
                                    # Note that we subtract `-1` to tt because the next state after taking action `a` at state `s` is in the SAME row of the df_trajectory
                                    # data frame where state `s` and action `a` were found (with location index stored in `idx`).
                                    position_of_particle_at_time_tt_after_state_s_action_a = df_trajectory['next_state'].iloc[idx + tt - 1]
                                    for x in self.states_of_interest:
                                        empirical_mean_actions[a] += int(position_of_particle_at_time_tt_after_state_s_action_a == x)
                                    n_particles_tt_after_state_s_actions[a] += 1
                                except IndexError:
                                    pass
                    if n_particles_tt_after_state_s == 0 and sum(n_particles_tt_after_state_s_actions) == 0:
                        # No particles were found that provide information about their position at time tt after visiting state s
                        # (this happens because either no particles visited state s or because the clock of all the particles that visited state s is still less than tt after having visited state s)
                        done_trajectories = True
                    else:
                        # Compute Phi(tt,x) as the PROPORTION of particles that are at state x at time tt after visiting state s
                        for x in self.states_of_interest:
                            # 1) Update Phi for the given start state s
                            # Note that Phi is updated ONLY if there is at least one sample for the estimate of such Phi
                            if n_particles_tt_after_state_s > 0:  # We don't want to update Phi if there is no sample for the given start state s
                                empirical_mean /= n_particles_tt_after_state_s
                                last_tt_inserted_in_phi[s][x] = update_phi_internal(self.dict_phi_for_state[s], tt, x, empirical_mean, last_tt_inserted_in_phi[s][x])

                                # When using an estimate of Phi(t,x) that does NOT depend on the start state-action for larger times
                                if False and self.burnin_time is not None and tt > self.burnin_time:
                                    # Replace the value just computed with an estimate of Phi(t,x) that does NOT depend on the start state, as we assume that stationarity has been reached
                                    # Note that we subtract -1 because if e.g. Phi['t'] = [0.0, 1.5, 2.3, 5.0] and if 2.3 <= tt < 5.0, the value of Phi at tt is Phi(t=2.3), because Phi is a piecewise constant function.
                                    # The side="right" parameter only affects when the searched value is in the list.
                                    # In the above example, np.searchsorted(phi, 1.5, side="left") gives 1 whereas with side="right" the result is 2
                                    # (the parameter indicates on which side of the list the tested value would be inserted if found already in the sorted list, including repetitions)
                                    idx_last_time_in_phi_smaller_than_or_equal_to_tt = np.searchsorted(self.dict_phi[x]['t'], tt, side="right") - 1
                                    assert idx_last_time_in_phi_smaller_than_or_equal_to_tt >= 0
                                    phi_value_at_tt = self.dict_phi[x]['Phi'].iloc[idx_last_time_in_phi_smaller_than_or_equal_to_tt]
                                    self.dict_phi_for_state[s][x].loc[tt] = [tt, phi_value_at_tt]

                            # 2) Update Phi for each possible start state at the given start state s
                            # Note that Phi is updated ONLY if there is at least one sample for the estimate of such Phi
                            for a in actions:
                                if n_particles_tt_after_state_s_actions[a] > 0:  # We don't want to update Phi if there is no sample for the given start state-action
                                    empirical_mean_actions[a] /= n_particles_tt_after_state_s_actions[a]
                                    last_tt_inserted_in_phi_actions[s][a][x] = update_phi_internal(self.dict_phi_for_state_action[s][a], tt, x, empirical_mean_actions[a], last_tt_inserted_in_phi_actions[s][a][x])

    def _update_phi_function_fixed_sample_size(self, start_state, start_action, t, state, next_state):
        "Updates the empirical mean phi(t, x; s) at the given time t, for x in intersection({state, next_state}, states_of_interest) for the given state s"
        self.dict_phi_for_state[start_state] = update_phi(self.env, self.N_for_start_state[start_state], t, self.dict_phi_for_state[start_state], state, next_state)
        self.dict_phi_for_state_action[start_state][start_action] = update_phi(self.env, self.N_for_start_state_action[start_state][start_action], t, self.dict_phi_for_state_action[start_state][start_action], state, next_state)

    def _update_particle_trajectory(self, env, t, state, action, next_state, reward):
        # Update the trajectory stored in the environment
        assert env.getStoreTrajectoryFlag(), "The trajectory must be stored in the environments representing the FV particles"
        env.update_trajectory(t, state, action, next_state, reward)

    def _compute_fv_deltas(self, t_surv, state_absorption, state, action):
        """
        Updates the FV contribution to the V(s) and Q(s,a) value functions. This update normally happens when an absorption is observed

        Arguments:
        t_surv: positive float
            Newly observed survival time that contributes to the survival probability P(T>t) estimation.
            When the learning criterion is the discounted learning criterion `t_surv` should be integer,
            as the underlying Markov process is assumed to be discrete-time.

        state_absorption: int
            Index representing the absorption state from which a bootstrap state value is retrieved
            as a contribution to the delta(V) and delta(Q) values computed by this method.

        state: int
            Index of the state whose value is updated.

        action: int
            Index of the action whose value is updated.

        Return: tuple
        Tuple with two elements:
        - delta_V for the given state to use in the update of the state value function at the given state.
        - delta_Q for the given state and action to use in the update of the action value function at the given state and action.
        """
        # Store the newly observed survival time in the dictionary where they are stored for the currently updated state and action
        # NOTE that, if the survival times are integer-valued, the newly observed survival time may already be present in the list.
        # When this is the case, the basic.insort() function called here to insert the survival time in order, inserts the value at the
        # end of all the occurrences. Therefore, idx_insert tells us the index of the smallest t among all observed survival times
        # that are larger than the t_surv just inserted.
        #print(f"[debug-compute_fv_deltas] s={state}, a={action}:")
        #print(f"[debug-compute_fv_deltas] Processing survival time {t_surv}...")
        basic.insort(self.dict_survival_times_for_state[state], t_surv)
        df_proba_surv_for_state = computing.compute_survival_probability(self.dict_survival_times_for_state[state], colnames=['t', 'P(T>t)'], right_continuous=False)
        basic.insort(self.dict_survival_times_for_state_action[state][action], t_surv)
        df_proba_surv_for_state_action = computing.compute_survival_probability(self.dict_survival_times_for_state_action[state][action], colnames=['t', 'P(T>t)'], right_continuous=False)

        # Uniformization constant which is used in scaling the discrete-time FV integral
        uniform_jump_rate = 1 #1 #self.getStateCounts()[state] #self.N
        # Initialize the state and action values of the currently analyzed state and action associated to the killed process,
        # which are updated below using the FV estimator.
        state_value_killed_process = 0.0
        action_value_killed_process = 0.0
        # Contributions to the state and action values for the killed process come from ALL states of interest, therefore we iterate on them
        for x in self.states_of_interest:
            # Align (in time) the two functions to multiply in the integrand, P(T>t; state, action) and Phi(t, x; state, action),
            # so that the same t is associated to each of the two function values being multiplied
            # (recall that x is the state of interest being considered at the current loop, while `state` and `action`
            # are the start state and action w.r.t. which all survival times giving rise to the estimate of P(T>t; state, action)
            # and to Phi(t, x; state, action) have been measured.

            # IMPORTANT: In the calculation of the FV integral, we need to DISCARD the first record of the merged data frame because
            # the FV summation formula (giving rise to the FV "integral") starts at t=1, NOT at t=0. That is, there is NO contribution from P(T>t) and Phi(t,x) at t=0.
            # (In fact, t is a sample of the survival time random variable when the process starts at a given state-action where the state is OUTSIDE the absorption set A,
            # therefore this survival time can never be 0.)
            # Note also that, we are safe in discarding the first record of the merged data frame because the second record will ALWAYS start at 1, as all the time values
            # stored in the 't' column of Phi(t,x) are consecutive integer values as asserted below.
            # However, even if they were not consecutive integer values, it might well be the case that starting to sum from the second record would still work because
            # we can safely ignore the contributions from t=1 until the time 't' recorded in the second record because they are ZERO if such 't' value is > 1.
            # In fact, if 't' in the second record is > 1, it means that the starting `state` giving rise to the values of Phi(t, x; state, action) is NOT equal to x;
            # if `state` were equal to x, then the second record in Phi(t,x) would have t=1 because Phi(t,x) would have been updated as soon as one of the particles
            # --included in the sample of particles starting at x-- moves. *** This ACTUALLY ASSUMES that when the particle moves, it CHANGES state
            # if this were NOT the case, we would need to assure that a new record in Phi(t,x) is added even if the particle that moved does not change state
            # (which may well happen if, e.g., the particle tried to move against an obstacle which in the end prevented it from moving). ***
            # In any case, as stated above, the current implementation of Phi(t, x; state, action) estimation guarantees that all times stored in 't' are CONSECUTIVE integers,
            # and this is because the implementation does NOT check whether the value of Phi would change before computing it (as is the case with the calculation of Phi(t,x)
            # that is not a function of the start state and action). Note that we do NOT do this check because the state-action dependent Phi can be updated at ANY time, even
            # at a time whose Phi value has already been computed before, and this is the case because the t values on which Phi is updated are NOT always increasing, as is the
            # case with the Phi(t,x) that is independent of the start state and action --because the measurement of such time DEPENDS on the start state and action.

            # Contribution to the state value
            assert all(np.diff(self.dict_phi_for_state[state][x]['t']) == 1), f"The times 't' stored in Phi(t,x={x}, s={state}) must be CONSECUTIVE integers:\n{self.dict_phi_for_state[state][x]}"
            df_phi_proba_surv_for_state = merge_proba_survival_and_phi(df_proba_surv_for_state, self.dict_phi_for_state[state][x])
            state_value_killed_process += compute_fv_integral(df_phi_proba_surv_for_state.iloc[1:], reward=self.env.getReward(x), interval_size=1/uniform_jump_rate, discount_factor=self.gamma)

            # Contribution to the action value
            assert all(np.diff(self.dict_phi_for_state_action[state][action][x]['t']) == 1), f"The times 't' stored in Phi(t,x={x}; s={state}, a={action}) must be CONSECUTIVE integers:\n{self.dict_phi_for_state_action[state][action][x]}"
            df_phi_proba_surv_for_state_action = merge_proba_survival_and_phi(df_proba_surv_for_state_action, self.dict_phi_for_state_action[state][action][x])
            action_value_killed_process += compute_fv_integral(df_phi_proba_surv_for_state_action.iloc[1:], reward=self.env.getReward(x), interval_size=1/uniform_jump_rate, discount_factor=self.gamma)

            ind_gt0 = (df_phi_proba_surv_for_state['P(T>t)'] > 0.0) & (df_phi_proba_surv_for_state['Phi'] > 0.0) & (df_phi_proba_surv_for_state['dt'] > 0.0)
            if np.sum(ind_gt0) > 0:
                print(f"[debug-compute_fv_deltas] s={state}, a={action}:")
                # Showing the values of P(T>t) and Phi may be generate too long lines
                #print(f"[debug-compute_fv_deltas] P(T>t; s):\n{df_proba_surv_for_state}")
                #print(f"[debug-compute_fv_deltas] Phi:\n{self.dict_phi_for_state[state][x]}")
                print(f"[debug-compute_fv_deltas] f(t,x={x}) = P(T>t; s) * Phi(t,x):\n{df_phi_proba_surv_for_state.loc[ind_gt0, :]}")
                print( "[debug-compute_fv_deltas] Integral(f(t,x={})) = {}".format(x, state_value_killed_process))
                print(f"[debug-compute_fv_deltas] alpha[s={state}] = {self.getAlphaForState(state)}")
                print("")
                if False and state == 3:
                    # Make an interactive plot of the survival probability and Phi functions that is updated every time these functions are updated for ANY start state
                    import matplotlib.pyplot as plt
                    fig = plt.figure(99)
                    print(f"Interactive: {plt.isinteractive()}")
                    ax = fig.gca()
                    ax.step(df_phi_proba_surv_for_state['t'], df_phi_proba_surv_for_state['P(T>t)'], color="blue", where='post')
                    ax.step(df_phi_proba_surv_for_state['t'], df_phi_proba_surv_for_state['Phi'], color="red", where='post')
                    ax.step(df_phi_proba_surv_for_state['t'], self.gamma**df_phi_proba_surv_for_state['t']*df_phi_proba_surv_for_state['P(T>t)']*df_phi_proba_surv_for_state['Phi'], color="green", where='post')
                    ax.set_title(f"P(T>t; state) (blue) and Phi(t,x; state) (red), gamma*P*Phi (green) for start state = {state}, x = {x}\nAbsorption state: {state_absorption}, Survival time = {t_surv}")
                    plt.draw()
                    plt.pause(0.01) # Need this pause() call with a positive argument in order for the plot to be drawn with plt.draw()!!

        #-- Compute the delta values to be used for the state value and action value functions update
        # The following discount has only an effect when self.gamma < 1
        discount_factor = self.gamma**t_surv
        #print(f"[debug] Discount factor applied to the bootstrapping value of the absorption state s={state_absorption}: {discount_factor} (t_surv={t_surv})\n")
        value_at_absorption_state = discount_factor * self.V.getValue(state_absorption)
        self.n_expected_discounted_state_value_at_absorption += 1
        self.expected_discounted_state_value_at_absorption += (value_at_absorption_state - self.expected_discounted_state_value_at_absorption) / self.n_expected_discounted_state_value_at_absorption
        #print("*** Updated expected discounted V(y): new value = {}, E = {} (n={})".format(value_at_absorption_state, self.expected_discounted_state_value_at_absorption, self.n_expected_discounted_state_value_at_absorption))

        # Initialize the delta's to compute to 0
        # The actual values are only computed when we receive a signal from the rare rewards outside A.
        delta_V = delta_Q = 0.0
        if state_value_killed_process != 0:
            #print("*** --> Proportion of the value of the killed process: {:.3f}%".format(self.expected_discounted_state_value_at_absorption / state_value_killed_process * 100))
            delta_V = state_value_killed_process  + self.expected_discounted_state_value_at_absorption - self.V.getValue(state)
            #print("*** --> V = {:.3f}, delta(V) = {:.3f} ({:.3f}%)".format(self.V.getValue(state), delta_V, delta_V / min(1, self.V.getValue(state))*100))
        if action_value_killed_process != 0:
            delta_Q = action_value_killed_process + self.expected_discounted_state_value_at_absorption - self.Q.getValue(state, action)

        return delta_V, delta_Q, state_value_killed_process, action_value_killed_process

    def _compute_fv_deltas_fixed_sample_size(self, t_surv, state_absorption, state, action):
        """
        Updates the FV contribution to the V(s) and Q(s,a) value functions based on the FV particles group that started at (s,a).
        This update normally happens when an absorption is observed

        Arguments:
        t_surv: positive float
            Newly observed survival time that contributes to the survival probability P(T>t) estimation.
            When the learning criterion is the discounted learning criterion `t_surv` should be integer,
            as the underlying Markov process is assumed to be discrete-time.

        state_absorption: int
            Index representing the absorption state from which a bootstrap state value is retrieved
            as a contribution to the delta(V) and delta(Q) values computed by this method.

        state: int
            Index of the state whose value is updated.

        action: int
            Index of the action whose value is updated.

        Return: tuple
        Tuple with two elements:
        - delta_V for the given state to use in the update of the state value function at the given state.
        - delta_Q for the given state and action to use in the update of the action value function at the given state and action.
        """
        # Store the newly observed survival time in the dictionary where they are stored for the currently updated state and action
        #  (this is used when updating the FV integral, normally by _update_integral_fixed_sample_size())
        # NOTE that, if the survival times are integer-valued, the newly observed survival time may already be present in the list.
        #print(f"[debug-compute_fv_deltas_fixed_sample_size] s={state}, a={action}:")
        #print(f"[debug-compute_fv_deltas_fixed_sample_size] Processing survival time {t_surv}...")
        self.dict_survival_times_for_state[state] += [t_surv]
        self.dict_survival_times_for_state_action[state][action] += [t_surv]
        state_value_killed_process = self._update_integral_fixed_sample_size(state)
        if state not in self.env.getTerminalStates():
            assert action >= 0, f"The start action must be a valid start action, i.e. there must be at least one particle that started at that action (s={state}, a={action})"
            # Note: for terminal states, the Q values are the same for all actions and equal to V(s)
            action_value_killed_process = self._update_integral_fixed_sample_size_action(state, action)
        else:
            action_value_killed_process = state_value_killed_process

        if False:
            print(f"[debug-compute_fv_deltas_fixed_sample_size] s={state}, a={action}:")
            # Showing the values of P(T>t) and Phi may be generate too long lines
            print(f"[debug-compute_fv_deltas_fixed_sample_size] Survival times:\n{self.dict_survival_times_for_state[state]}")
            for x in self.states_of_interest:
                print(f"[debug-compute_fv_deltas_fixed_sample_size] Phi(t,x={x}):\n{self.dict_phi_for_state[state][x]}")
                print( "[debug-compute_fv_deltas_fixed_sample_size] Integral(x={}) = {:.3f}".format(x, state_value_killed_process))
            print(f"[debug-compute_fv_deltas_fixed_sample_size] alpha[s={state}] = {self.getAlphaForState(state)}")
            print("")

        #-- Compute the delta values to be used for the state value and action value functions update
        # The following discount has only an effect when self.gamma < 1
        discount_factor = self.gamma**t_surv
        #print(f"[debug] Discount factor applied to the bootstrapping value of the absorption state s={state_absorption}: {discount_factor} (t_surv={t_surv})\n")
        value_at_absorption_state = discount_factor * self.V.getValue(state_absorption)
        self.n_expected_discounted_state_value_at_absorption += 1
        self.expected_discounted_state_value_at_absorption += (value_at_absorption_state - self.expected_discounted_state_value_at_absorption) / self.n_expected_discounted_state_value_at_absorption
        #print("*** Updated expected discounted V(y): new value = {}, E = {} (n={})".format(value_at_absorption_state, self.expected_discounted_state_value_at_absorption, self.n_expected_discounted_state_value_at_absorption))

        # Initialize the delta's to compute to 0
        # The actual values are only computed when we receive a signal from the rare rewards outside A.
        delta_V = delta_Q = 0.0
        if True or state_value_killed_process != 0:
            #print("*** --> Proportion of the value of the killed process: {:.3f}%".format(self.expected_discounted_state_value_at_absorption / state_value_killed_process * 100))
            delta_V = state_value_killed_process  + self.expected_discounted_state_value_at_absorption - self.V.getValue(state)
            #print("*** --> V = {:.3f}, delta(V) = {:.3f} ({:.3f}%)".format(self.V.getValue(state), delta_V, delta_V / min(1, self.V.getValue(state))*100))
        if True or action_value_killed_process != 0:
            delta_Q = action_value_killed_process + self.expected_discounted_state_value_at_absorption - self.Q.getValue(state, action)

        return delta_V, delta_Q, state_value_killed_process, action_value_killed_process

    def _update_absorption_times_and_phi(self, t_absorption: int):
        """
        Updates the list of observed absorption times and the dictionary containing the index (normally self.dict_last_indices_phi_prior_to_absorption_times),
        for each state of interest x and for each observed absorption time,
        that signals the latest entry in dict_phi[x] whose measurement time occurs before the respective
        absorption time (where "respective" means: "stored at the same index position (of the attribute
        storing all observed absorption times, normally self.absorption_times) as the index position at which the information in the
        aforementioned dictionary is stored").

        This index information is used when computing the Phi contribution to the Fv integral update
        at the occurrence of a new absorption time. And because the update of the FV integral happens
        at an absorption time, only the index value stored at the *previous to last position* of the list
        stored in the dictionary entry x (indicating the state of interest for which the FV integral is
        being updated) is RELEVANT. Note that the LAST index stored in the list is NOT relevant (at the
        time of occurrence of a new absorption time) because that index coincides with the number of rows
        present in dict_phi[x] (because this method also updates dict_phi[x] when no change in Phi
        is observed between the previous absorption time and the latest absorption time) (an assertion is
        in place in the _update_phi_contribution() method that checks this).

        Therefore, if we wanted to save storage, we could keep the list storing this index information
        just the two latest values (but we don't do this at this time, due to e.g. debugging purposes).

        Ex:
        self.absorption_times = [0, 3, 8, 13]
        self.dict_phi[x=9] = {'t': [0, 2, 4, 5, 7, 12], 'Phi': [0, 0.1, 0.0, 0.1, 0.2, 0.2]}
        self.dict_last_indices_phi_prior_to_absorption_times[x=9] = [0, 1, 4, 5]

        Note that self.dict_last_indices_phi_prior_to_absorption_times[x=9] has *the same length* as
        self.absorption_times, and this is because their indices are linked. So, for example, the value
        at index 2 of self.dict_last_indices_phi_prior_to_absorption_times[x=9] says that the last time
        that Phi(x=9) was updated prior to the absorption time stored at index 2 in self.absorption_times
        (absorption time = 8) was the time in self.dict_phi[x=9] at index position 4 in entry 't',
        i.e. at time t = 7 (at which time the value of Phi(x=9) was updated from 0.0 to 0.1).
        """
        assert t_absorption > 0, f"The absorption time is positive (in particular >= 1): {t_absorption}"
        assert len(self.absorption_times) > 0
        previous_absorption_time = self.absorption_times[-1]
        for x in self.states_of_interest:
            assert self.dict_phi[x].shape[0] > 0
            latest_t_value_stored_in_phi_x = self.dict_phi[x]['t'].iloc[-1]
            if latest_t_value_stored_in_phi_x <= previous_absorption_time:
                # There was no change on the empirical distribution at x between the previous absorption time and the current absorption time
                # => Add an new row entry in Phi(x) in order to indicate that Phi(x) remained constant in the
                # previous inter-absorption interval, in order to facilitate the calculation of the update
                # of the Phi contribution to the FV integral typically done by _update_phi_contribution().
                # The new entry will have t_absorption - TIME_RESOLUTION (i.e. a value strictly smaller than t_absorption)
                # so that the assigned time value is part of the latest inter-absorption interval, i.e. the interval that ends at the currently observed absorption time.
                # NOTE that t_absorption - TIME_RESOLUTION is *lower bounded* by the previous absorption time, and the bound is materialized normally when
                # the current absorption time *coincides* with the previous absorption time (in the current implementation of the FV simulation,
                # this happens when, precisely at the maximum simulation time (e.g. max_time_steps = 5000), a particle is absorbed while some particles are left unabsorbed,
                # a situation that triggers their forced absorption at the EXACT same time of the last particle absorption (e.g. at t = 5000).
                self.dict_phi[x] = pd.concat([self.dict_phi[x],
                                              pd.DataFrame({'t': [max(previous_absorption_time, t_absorption - self.TIME_RESOLUTION)],
                                                            'Phi': [self.dict_phi[x]['Phi'].iloc[-1]]},
                                                           index=[self.dict_phi[x].shape[0]])],
                                             axis=0)
            # Add to the corresponding dictionary the last index in Phi(x) that contributes to the latest/current inter-absorption interval
            self.dict_last_indices_phi_prior_to_absorption_times[x] += [ self.dict_phi[x].shape[0] - 1 ]

        # Store the current absorption time
        self.absorption_times += [t_absorption]

    def _update_phi_contribution(self, t_absorption: int):
        """
        Updates the Phi contribution to the integral coming from the latest interval of events
        between the previous and the current absorption time.

        This update is useful only when all survival times contributing to the P(T>t) estimator occur in increasing order.
        """
        assert len(self.absorption_times) > 1, "There are at least 2 absorption times stored in the list of absorption times (meaning there has been at least one observed absorption time)"
        t_absorption_prev = self.absorption_times[-2]
        for x in self.states_of_interest:
            assert self.dict_phi[x].shape[0] > 0, f"The Phi data frame has been initialized for state = {x}"
            assert len(self.dict_last_indices_phi_prior_to_absorption_times[x]) > 1, "At least 2 intervals of events have been observed before the current absorption time"

            phi_sum_latest_interval_x = 0.0    # "latest" refers to the currently evaluated interval of events observed between the previous absorption time and the current absorption time
            idx_first_phi_x_measurement_in_latest_events_interval = self.dict_last_indices_phi_prior_to_absorption_times[x][-2] + 1
            idx_last_phi_x_measurement_in_latest_events_interval = self.dict_last_indices_phi_prior_to_absorption_times[x][-1]
            assert idx_first_phi_x_measurement_in_latest_events_interval > 0
            assert idx_last_phi_x_measurement_in_latest_events_interval == self.dict_phi[x].shape[0] - 1, \
                f"The value of idx_last_phi_x_measurement_in_latest_events_interval ({idx_last_phi_x_measurement_in_latest_events_interval})" \
                f" must be equal to the number of rows stored in self.dict_phi[x={x}] minus 1 ({self.dict_phi[x].shape[0]-1})"
            # Compute the contribution to the Phi sum from the Phi(x) updates taking place in the latest inter-absorption interval
            # NOTE that these contributions could be none, when no updates have happened, and this case is signalled by the situation
            # where idx_first_phi_x_measurement_in_latest_events_interval = idx_last_phi_x_measurement_in_latest_events_interval + 1
            for idx in range(idx_first_phi_x_measurement_in_latest_events_interval, idx_last_phi_x_measurement_in_latest_events_interval + 1):
                t1 = self.dict_phi[x]['t'].iloc[idx-1] if idx > idx_first_phi_x_measurement_in_latest_events_interval else t_absorption_prev
                t2 = self.dict_phi[x]['t'].iloc[idx]
                _deltat = t2 - t1
                assert _deltat >= 0, f"t1={t1}, t2={t2}, deltat={_deltat}"
                # The value of Phi(x) contributing to each interval deltat = t(i) - t(i-1) is the value at t(i-1), NOT the value at t(i)
                # because at t(i) is when Phi(x) changes value (following the change of a step function, which is what Phi(x) is, a step function)
                _phi = self.dict_phi[x]['Phi'].iloc[idx-1]
                #print(f"delta={_deltat}, Phi={_phi}")
                phi_sum_latest_interval_x += _phi * _deltat
            # Contribution to the last micro-interval, [t(last), t_absorption], where t(last) is the time at the last change in Phi recorded at Phi(x)'s row index `idx_last_phi_x_measurement_in_latest_events_interval`
            _deltat = t_absorption - self.dict_phi[x]['t'].iloc[idx]
            _phi = self.dict_phi[x]['Phi'].iloc[idx]
            #print(f"delta={_deltat}, Phi={_phi}")
            phi_sum_latest_interval_x += _phi * _deltat

            # Store this value in the dictionary of Phi sums
            self.dict_phi_sum[x] = pd.concat([self.dict_phi_sum[x],
                                              pd.DataFrame({'t': [t_absorption], 'Phi': [phi_sum_latest_interval_x]}, index=[self.dict_phi_sum[x].shape[0]])],
                                             axis=0)

    def _update_integral(self, fixed_sample_size=True):
        """
        Updates, for each x among the states of interest, the FV integral by computing the change in the area below the curve P(T>t) * Phi(t,x)
        generated by the observation of a new survival time, which is assumed to have already been stored in attribute self.absorption_times.

        This update assumes that all survival times contributing to the P(T>t) estimator occur in increasing order.

        The attribute self.dict_integral is updated with the updated integral value for each x in self.states_of_interest.

        fixed_sample_size: (opt) bool
            Whether the integral should be updated based on N survival times contributing to the estimation of the survival probability P(T>t).
            This is the case when we only consider the FIRST survival time of each particle as contribution to the estimation of P(T>t).
            default: True
        """
        # Number of absorption times observed so far (assumed observed in increasing order)
        n = len(self.absorption_times) - 1  # -1 because the first absorption time in the list is dummy, and is set to 0
        if fixed_sample_size:
            for x in self.states_of_interest:
                phi_contribution_from_latest_interval = self.dict_phi_sum[x]['Phi'].iloc[-1]
                self.dict_integral[x] += self.env.getReward(x) * (1 - (n - 1) / self.N) * phi_contribution_from_latest_interval
        else:
            for x in self.states_of_interest:
                first_n_minus1_phi_contributions = self.dict_phi_sum[x]['Phi'].iloc[1:-1]   # We start at index 1 because index 0 contains the dummy value 0, which does not really correspond to a phi contribution to the integral.
                phi_contribution_up_to_n_minus_1 = 1/(n*(n-1)) * np.sum( np.arange(n-1) * first_n_minus1_phi_contributions ) if n > 1 else 0.0
                phi_contribution_from_interval_n = 1/n * self.dict_phi_sum[x]['Phi'].iloc[-1]
                self.dict_integral[x] += self.env.getReward(x) * (phi_contribution_up_to_n_minus_1 + phi_contribution_from_interval_n)

    def _update_integral_fixed_sample_size(self, state):
        """
        Updates, for each x in the states of interest, the FV integral by adding the new area below the curve P(T>t) * Phi(t,x) that contributes
        to the integral after the observation of a new survival time, which is assumed to have already been stored in attribute self.absorption_times.

        This update is suitable when the sample size on which the survival times are observed is known in advance and fixed, in which case it is given by:
            delta(Integral) = (1 - (n-1)/N) * Integral_{t(n-1), t(n)} { Phi(t,x) dt }
        where n is the number of non-zero survival times observed so far (observed in increasing order), N is the number of FV particles,
        t(n-1) is the (n-1)-th observed survival time and t(n) is the last n-th observed survival time.

        This update assumes that all survival times contributing to the P(T>t) estimator occur in increasing order.

        Note that this update corresponds to the same update that would be computed for the **Kaplan-Meier estimator** of the survival function
        (but is MUCH simpler), and this is true because there are NO censoring times in between observed consecutive survival times.
        In fact, the survival time for ALL particles will be eventually observed.
        For more details on this equivalence, see my notes in the blue notebook "Univ. Lorraine" dated 28-Jan-2024.

        The attribute self.dict_integral is updated with the updated integral value for each x in self.states_of_interest.

        Arguments:
        state: int
            Index of the start state for which the FV integral is updated.

        Return: float
        The updated FV integral (discrete sum) in the discounted (self.gamma < 1) or undiscounted setting (self.gamma = 1) computed as the sum of the
        r(x)*gamma^(t-1)*P(T>t)*Phi(t,x) contributions --for the start state given as input parameter-- on all states of interest x,
        where r(x) is the reward observed when visiting x.
        """
        # Number of absorption times observed so far (assumed observed in increasing order)
        n = len(self.dict_survival_times_for_state[state]) - 1  # -1 because the first survival time stored in the list of survival times is dummy, and is set to 0
        interval_size = 1 #1 / self.N_for_start_state[state] # This would be the adjustment to the FV integral because the observed times come from several Markov chains (partilces), as opposed to from just one Markov chain

        # Value of  the integral to return, which is updated in the loop below by the contribution from the rewards from ALL states of interest
        # (i.e. recall the expression for q(s,a) in Overleaf: sum_{x}{ r(x) * sum_{t>=1}{ gamma^(t-1) * P(T>=t; s,a) * Phi(t,x; s,a) } })
        fv_integral_over_all_states_of_interest = 0.0
        for x in self.states_of_interest:
            # Extract the part of Phi that should be summed as contribution to the update of the integral
            phi_for_state = self.dict_phi_for_state[state][x]
            idx_previous_absorption = list(phi_for_state['t']).index(self.dict_survival_times_for_state[state][-2])
            phi_to_sum = phi_for_state.iloc[idx_previous_absorption:].copy()
            if phi_to_sum['t'].iloc[0] == 0.0:
                # We should NOT include the contribution from the interval t in [0, 1) in the computation of the FV integral because the sum giving the FV integral starts at t = 1!
                # Since the first time where Phi changes from its value at t(0), may be LARGER than 1, i.e. t(1) could be > 1, we need to make sure that the interval [1, t(1)) is
                # considered in the sum. To this end we simply replace t = 0 in the first record of Phi with t = 1. If t(1) > 1, we are in business because Phi(1) = Phi(0), as
                # Phi did NOT change from t=0 to t=1 (o.w. t(1) would be 1); and if t(1) = 1, the value of 'dt' for the first record (computed below) will be 0, making the
                # value Phi(0) NOT contribute to the sum, which is what we want.
                # Ex 1:
                #   t =   [0, 3, 5]
                #   Phi = [1.0, 0.6, 0.8]
                #   dt =  [3, 2, 0]
                # will become:
                #   t =   [1, 3, 5]
                #   Phi = [1.0, 0.6, 0.8]
                #   dt =  [2, 2, 0]
                # therefore the first contribution will be Phi(t=1) = 1.0 which is constant in the interval [1, 3), hence the terms contributing from such interval will be:
                # gamma^(1-1)*Phi(t=1) + gamma^(2-1)*Phi(t=2) = 1*1.0 + gamma*1.0
                # which is the way to go.
                # Ex 2:
                #   t =   [0, 1, 3, 5]
                #   Phi = [1.0, 0.8, 0.6, 0.8]
                #   dt =  [1, 2, 2, 0]
                # will become:
                #   t =   [1, 1, 3, 5]
                #   Phi = [1.0, 0.8, 0.6, 0.8]
                #   dt =  [0, 2, 2, 0]
                # therefore, since dt = 0 between the first and second record, only the values of Phi(t=1) = 0.8 and Phi(t=3) = 0.6 will contribute to the FV integral,
                # which is the correct thing to do.
                phi_to_sum['t'].iloc[0] = 1
            assert len(phi_to_sum) >= 2, f"The values on which the Phi sum should be computed are at least 2, i.e. the value of Phi at the previous absorption time, and the value of Phi at the current absorption time: Phi at the last inter-absorption interval =\n{phi_to_sum}"
            # Compute the time interval between two consecutive time values stored in Phi(t,x), computed as dt = t(i+1) - t(i)
            # Note that 0.0 is the dt value assigned for the last interval starting at the survival time observed last, so that the Phi value associated to that interval
            # does NOT contribute to the updated integral, as such interval should contribute on the NEXT interval, when a new survival time is observed.
            phi_to_sum.loc[:, 'dt'] = np.r_[np.diff(phi_to_sum['t']), 0.0]

            # Compute the integral update coming from the last interval (between the two latest observed survival times)
            # Note that in the phi_to_sum data frame, the time 't' references the interval that is to its RIGHT and ends at the time 't'+'dt'.
            # and recall that the interval starting at the last observed survival time (for which there is a Phi value as well) is NOT summed
            # because it's dt value was set to 0 above (where this situation was also explained).
            if self.gamma == 1:
                discounted_phi_sum_in_interval_n = np.sum(phi_to_sum['Phi'] * phi_to_sum['dt'])
            else:
                # For more details about this calculation, see compute_fv_integral() in estimators/fv.py
                assert 0 < self.gamma < 1
                discounted_phi_sum_in_interval_n = 1 / (1 - self.gamma) * np.sum( self.gamma**(phi_to_sum['t'] - 1) * (1 - self.gamma**(phi_to_sum['dt'])) * phi_to_sum['Phi'] )
            contribution_from_interval_n = (1 - (n-1) / self.N_for_start_state[state]) * discounted_phi_sum_in_interval_n
            self.dict_integral_for_state[state][x] += interval_size * self.env.getReward(x) * contribution_from_interval_n
            fv_integral_over_all_states_of_interest += self.dict_integral_for_state[state][x]

        return fv_integral_over_all_states_of_interest

    def _update_integral_fixed_sample_size_action(self, state, action):
        """
        """
        # Number of absorption times observed so far (assumed observed in increasing order)
        n = len(self.dict_survival_times_for_state_action[state][action]) - 1  # -1 because the first survival time stored in the list of survival times is dummy, and is set to 0
        interval_size = 1 #1 / self.N_for_start_state_action[state][action] # This would be the adjustment to the FV integral because the observed times come from several Markov chains (partilces), as opposed to from just one Markov chain

        # Value of  the integral to return, which is updated in the loop below by the contribution from the rewards from ALL states of interest
        # (i.e. recall the expression for q(s,a) in Overleaf: sum_{x}{ r(x) * sum_{t>=1}{ gamma^(t-1) * P(T>=t; s,a) * Phi(t,x; s,a) } })
        fv_integral_over_all_states_of_interest = 0.0
        for x in self.states_of_interest:
            # Extract the part of Phi that should be summed as contribution to the update of the integral
            phi_for_state_action = self.dict_phi_for_state_action[state][action][x]
            idx_previous_absorption = list(phi_for_state_action['t']).index(self.dict_survival_times_for_state_action[state][action][-2])
            phi_to_sum = phi_for_state_action.iloc[idx_previous_absorption:].copy()
            if phi_to_sum['t'].iloc[0] == 0.0:
                phi_to_sum['t'].iloc[0] = 1
            assert len(phi_to_sum) >= 2, f"The values on which the Phi sum should be computed are at least 2, i.e. the value of Phi at the previous absorption time, and the value of Phi at the current absorption time: Phi at the last inter-absorption interval =\n{phi_to_sum}"
            # Compute the time interval between two consecutive time values stored in Phi(t,x), computed as dt = t(i+1) - t(i)
            # Note that 0.0 is the dt value assigned for the last interval starting at the survival time observed last, so that the Phi value associated to that interval
            # does NOT contribute to the updated integral, as such interval should contribute on the NEXT interval, when a new survival time is observed.
            phi_to_sum.loc[:, 'dt'] = np.r_[np.diff(phi_to_sum['t']), 0.0]

            # Compute the integral update coming from the last interval (between the two latest observed survival times)
            # Note that in the phi_to_sum data frame, the time 't' references the interval that is to its RIGHT and ends at the time 't'+'dt'.
            # and recall that the interval starting at the last observed survival time (for which there is a Phi value as well) is NOT summed
            # because it's dt value was set to 0 above (where this situation was also explained).
            if self.gamma == 1:
                discounted_phi_sum_in_interval_n = np.sum(phi_to_sum['Phi'] * phi_to_sum['dt'])
            else:
                # For more details about this calculation, see compute_fv_integral() in estimators/fv.py
                assert 0 < self.gamma < 1
                discounted_phi_sum_in_interval_n = 1 / (1 - self.gamma) * np.sum( self.gamma**(phi_to_sum['t'] - 1) * (1 - self.gamma**(phi_to_sum['dt'])) * phi_to_sum['Phi'] )
            contribution_from_interval_n = (1 - (n-1) / self.N_for_start_state[state]) * discounted_phi_sum_in_interval_n
            self.dict_integral_for_state_action[state][action][x] += interval_size * self.env.getReward(x) * contribution_from_interval_n
            fv_integral_over_all_states_of_interest += self.dict_integral_for_state_action[state][action][x]

        return fv_integral_over_all_states_of_interest

    #-- GETTERS
    def getNumParticles(self):
        return self.N

    def getStartStates(self):
        return self.start_states

    def getStartState(self, idx_particle):
        return self.start_states[idx_particle]

    def getStartActions(self):
        return self.start_actions

    def getStartAction(self, idx_particle):
        return self.start_actions[idx_particle]

    def getNumTimeStepsForExpectation(self):
        return self.T

    def getAbsorptionSet(self):
        return self.absorption_set

    def getActivationSet(self):
        return self.activation_set

    def getActiveSet(self):
        return self.active_set

    def getIntegral(self):
        return self.dict_integral

    def getProbasStationaryStartStateET(self):
        return self.probas_stationary_start_state_et

    def getProbasStationaryStartStateFV(self):
        return self.probas_stationary_start_state_fv

    #-- SETTERS
    def setStartStateAction(self, idx_particle, state, action):
        self.start_states[idx_particle] = state
        self.start_actions[idx_particle] = action
        self.start_state_actions[idx_particle] = (state, action)
        self.N_for_start_state[state] += 1
        self.N_for_start_action[action] += 1
        self.N_for_start_state_action[state][action] += 1

    def setProbasStationaryStartStateET(self, dict_proba):
        self.probas_stationary_start_state_et = dict_proba

    def setProbasStationaryStartStateFV(self, dict_proba):
        self.probas_stationary_start_state_fv = dict_proba
