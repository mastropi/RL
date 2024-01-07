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

    probas_stationary_start_state_absorption: (opt) dict
        Dictionary containing the probability for the selection of the start state to be used to estimate
        the denominator of the FV estimator of expectations, namely the expected reabsorption cycle time, E(T_A).
        When None, a random distribution is used to select the start state.
        default: None

    probas_stationary_start_state_activation=None,
        Dictionary containing the probability for the selection of the start state to be used for
        the initialization of the N particles of the FV process, which are used to estimate the survival
        probability P(T>t) and the empirical distribution Phi(x,t) that estimates the occupation
        probability of each state x of interest conditional to the process not being absorbed.
        When None, a random distribution is used to select the N start states.
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

    burnin_time: (opt) int or None
        Burn-in time by which the survival probability is shifted when aligning its measurement times
        with the measurement times of the estimate of the empirical mean Phi(t,x), which can be used
        to compute the FV integral using the times in Phi(t,x) where Phi is assumed to have reached stationarity.
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
        default: 1E-9
    """

    def __init__(self, env, N: int, T: int, absorption_set: set, activation_set: set,
                 states_of_interest: set=None,
                 probas_stationary_start_state_absorption: dict=None,
                 probas_stationary_start_state_activation: dict=None,
                 criterion=LearningCriterion.AVERAGE,
                 alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 burnin_time=0, TIME_RESOLUTION=1E-9,
                 debug=False):
        super().__init__(env, criterion=criterion, task=LearningTask.CONTINUING, alpha=alpha,  gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha,
                         alpha_update_type=alpha_update_type, adjust_alpha_by_episode=adjust_alpha_by_episode,
                         alpha_min=alpha_min,
                         store_history_over_all_episodes=True,  # We set this to True because FV is a CONTINUING learning task
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed, debug=debug)

        # Fleming-Viot simulation parameters
        self.N = N          # Number of particles in the FV simulation used to estimate the QSD Phi(t,x) for every state x at discrete time t
        self.T = T          # Maximum number of time steps that are allowed to be run in each episode of the single Markov chain that estimates the expected reabsorption time E(T_A)
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set ({}).".format(type(absorption_set)))
        if not isinstance(activation_set, set):
            raise ValueError("Parameter `activation_set` must be a set ({}).".format(type(activation_set)))
        self.absorption_set = absorption_set
        self.activation_set = activation_set
        self.states_of_interest = self.env.getTerminalStates() if states_of_interest is None else states_of_interest
        # Dictionary that stores the optional stationary probability of the start states of the single chain simulation
        # and of the N-particle Fleming-Viot process.
        # If None, uniformly random distributions are used.
        self.probas_stationary_start_state_absorption = probas_stationary_start_state_absorption
        self.probas_stationary_start_state_activation = probas_stationary_start_state_activation

        # Note: The average reward that is estimated by the FV learner is already stored in the GenericLearner class
        # which is the superclass of the Learner class which in turn is the superclass of the LeaTDLambda class
        # from which this LeaFV class inherits.

        # Time resolution for the underlying Markov process on which Fleming-Viot acts
        # If the process is discrete-time, the time resolution should be set to 1.
        self.TIME_RESOLUTION = TIME_RESOLUTION

        #-- Attributes that are defined in reset()
        self.absorption_times = None
        # Dictionary indicating the indices in self.dict_phi (for each state of interest x)
        # that inform the last Phi(x) row index whose time 't' is BEHIND the respective absorption time
        # stored in self.absorption_times.
        # Note that, for each x, dict_last_indices_phi_prior_to_absorption_times[x] contains
        # as many indices as the length of self.absorption_times).
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
        # Survival times for each state and for each state and action, i.e. measured from each time the state (or state and action) is visited
        # during a trajectory just before an absorption (leading to a survival time) happens.
        self.dict_survival_times_for_state = None
        self.dict_survival_times_for_state_action = None
        # FV integral value for each state of interest x (updated every time a new absorption is observed)
        self.dict_integral = None

        # Burn-in time that should be waited to consider that Phi(t,x) has achieved stationarity,
        # and thus independent of the starting state and action where the particles contributing to its estimation
        # can be considered to have reached stationary behaviour.
        # In fact, when computing the integral of P(T>t; s,a) * Phi(t,x) for each state of interest x,
        # we are assuming stationarity when writing Phi(t,x) instead of Phi(t,x; s,a), i.e. Phi(t,x) is assumed to NOT depend on
        # the "start" state-actions (s,a)!
        # This value is subtracted from the t values in the entries of Phi(t,x) when aligning P(T>t; s,a) with Phi(t,x)
        # (see estimators.fv.merge_proba_survival_and_phi() for more details).
        self.burnin_time = self.N*2 if burnin_time is None else burnin_time

        self.reset()

    def reset(self, reset_episode=False, reset_value_functions=False):
        super().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)

        # Reset the (integer) absorption times
        # We initialize the list of absorption times to 0 to make the update of the Phi contribution to the FV integral
        # easier to carry out, with less IF conditions that check whether a particle has already been absorbed.
        self.absorption_times = [0]

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

        # Survival probability functions are defined for each state in the set of active states (i.e. all the states that are NOT in the absorption set A)
        self.dict_survival_times_for_state = dict()
        self.dict_survival_times_for_state_action = dict()
        active_set = set(self.env.getAllStates()).difference(self.absorption_set)
        for s in active_set:
            # Survival functions for each state and for each state-action
            self.dict_survival_times_for_state[s] = [0.0]
            self.dict_survival_times_for_state_action[s] = dict()
            for a in range(self.env.getNumActions()):
                self.dict_survival_times_for_state_action[s][a] = [0.0]

    def learn(self, t, state, action, next_state, reward, done, info, env=None, update_phi=False):
        """
        Learns the value functions using TD learning and updates the particle's trajectory in the environment representing the particle

        Arguments:
        env: (opt) Environment
            Environment (particle) whose trajectory history should be updated with the new transition.
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

        if env is not None:
            assert env.store_trajectory, "The environment associated to the FV particle on which the value functions learning is carried out must have stored the observed trajectory"

            if update_phi:
                # Update the empirical distribution of both state and next_state when any of them is part of the set of states of interest.
                # This should ONLY happen when the particle (env) transitioning is an FV particle, as opposed to e.g. a particle that can freely explore the whole environment.
                # Note that `t` is the ABSOLUTE time, i.e. the time elapsed since the start of the FV simulation which is
                # independent of any survival time, which on the contrary is relative to the start position (and action) of the particle
                # that is eventually absorbed (i.e. the survival time is measured w.r.t. the time at which the particle was
                # at each state s visited by the particle's trajectory and taking action `a`).
                # Note however that at the moment of computing the FV integral, there will be an alignment between the
                # survival time associated to the survival probability of a particle starting at state s and action `a`
                # and the time t stored in the Phi(t,x) function considered on the state of interest x.
                self._update_phi(t, state, next_state)

            # Store the trajectory in the particle (environment) that has evolved
            # (so that we are able to retrieve its history when learning the value functions by the FV estimator)
            self._update_particle_trajectory(env, t, state, action, next_state, reward)
            #print(f"[debug-learn] Trajectory of particle just moved UPDATED:\n{env.getTrajectory()}")

    def _update_phi(self, t, state, next_state):
        "Updates the empirical mean phi(t, x) for x in intersection({state, next_state}, states_of_interest)"
        self.dict_phi = update_phi(self.env, self.N, t, self.dict_phi, state, next_state)

    def learn_at_absorption(self, env, t, state_absorption, next_state):
        """
        Learns the value functions at the time of the particle's absorption for EVERY state and action
        the particle visited prior to absorption, using:
        - the FV estimator of the value function of the killed process
        - the n-step bootstrap estimator of the value function, where n is the number of steps taken
        until absorption as measured from the time the particle was at the state and action whose value
        is now updated).

        The method also updates Phi(t,x) for every state of interest x as a result of reactivation
        from the absorption state `state_absorption` to `next_state`.

        Arguments:
        env: Environment (particle)
            Environment associated to the absorbed particle.
            This is used to extract the states and actions visited by the particle prior to absorption
            alongside the times they were visited at.

        t: positive float
            Current absolute time of the simulation, which is updated whenever ANY particle moves.
            It is the time JUST BEFORE the absorption happens, i.e. before the Markov process moves
            from state S(t) to the absorption state at t+1. Therefore the absorption time is actually t+1.
            This value is used to update Phi(t,x) and to update the self.absorption_times list that keeps track of
            all the SYSTEM absorption times (in this case it adds the value t+1 to the list).
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
        self._update_phi(t, state_absorption, next_state)

        #-- Learn the value function using the FV estimator
        # This means: update the value functions for all states and actions visited by the particle before absorption

        # Compute the number of steps it took the particle to become absorbed
        # Note that, in principle, we could use either this as absorption time of the particle OR the absolute time plus 1 (t + 1),
        # (i.e. either the particle's clock OR the FV system's clock) to compute the survival time for each start state s.
        # Note that the scaling of the FV integral is different depending on which of the two approaches we use for the absorption time:
        # - when using the particle's clock, the scaling is 1 because we are considering each particle separately and its change rate is 1 change per unit time.
        # - when using the FV system's clock, the scaling is 1/N because the change rate of the system is N per unit time, which defines the uniformization
        # constant that discretizes the originally continuous-time FV process.
        # Note that the particle's clock is given simply by the INDEX of the data frame storing the particle's trajectory.
        df_trajectory = env.getTrajectory()
        t_particle_abs = df_trajectory.shape[0]
        assert t_particle_abs > 0, f"The absorption time of the particle is > 0 ({t_particle_abs})"
        print(f"[debug-learn_at_absorption] Trajectory of the absorbed particle:\n{env.getTrajectory()}\n")
        for row in df_trajectory.iterrows():
            # Internal time of the absorbed particle (it ALWAYS takes the values 0, 1, 2, ...)
            # (which is NOT the simulation time because the simulation time increases by 1 at the movement of ANY particle)
            t_particle = row[0]
            row_data = row[1]

            # The absorption time is +1 the input time t given as parameter because this input time t is the time BEFORE the update of the state,
            # i.e. the time at which the Markov process was at the state BEFORE being absorbed.
            self.absorption_times += [t + 1]
            # Compute the survival time (i.e. until killing) using as origin the time when the particle visited the state and action
            # currently retrieved from the particle's trajectory
            t_surv = t_particle_abs - t_particle   # Use this if using the particle's clock to compute the survival time --> in this case the value of uniform_jump_rate in _compute_fv_deltas() should be 1
            #t_surv = (t + 1) - row_data['time']     # Use this if using the system's clock to compute the survival time --> in this case the value of uniform_jump_rate in _compute_fv_deltas() should be self.N (the number of particles)
            state = row_data['state']
            action = row_data['action']
            assert state in self.dict_survival_times_for_state.keys(), f"All states visited by the particle prior to absorption need to be in the active set of states (s={state})"
            delta_V, delta_Q, state_value_killed_process, action_value_killed_process = self._compute_fv_deltas(t_surv, state_absorption, state, action)

            # print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_over_all_episodes[state], self._alphas[state]))
            self._updateV(delta_V)
            self._updateQ(delta_Q)

            # Update alpha for the next iteration for the "by state counts" update
            # NOTE that we only update alpha when the value V(state) was actually updated (i.e. delta(V) != 0),
            # o.w. we don't want to decrease alpha when no signal was received from the rare rewards outside A.
            #print("Learn: state = {}, next_state = {}, done = {}".format(state, next_state, done))
            if delta_V != 0.0:
                self._update_alphas(state)
            # TODO: (2024/01/03) ALSO update the alphas by state-action (to be defined in the GenericLearner class first)

        # Reset the trajectory of the particle for the next absorption event
        env.reset_trajectory()

    def _update_particle_trajectory(self, env, t, state, action, next_state, reward):
        # Update the trajectory stored in the environment
        assert env.store_trajectory, "The trajectory must be stored in the environments representing the FV particles"
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
            as a contribution to the delta V and delta Q values computed by this method.

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
        df_proba_surv_for_state = computing.compute_survival_probability(self.dict_survival_times_for_state[state])
        basic.insort(self.dict_survival_times_for_state_action[state][action], t_surv)
        df_proba_surv_for_state_action = computing.compute_survival_probability(self.dict_survival_times_for_state_action[state][action])

        # Uniformization constant which is used in scaling the discrete-time FV integral
        uniform_jump_rate = 1 #1 #self.getStateCounts()[state] #self.N
        # Initialize the state and action values of the currently analyzed state and action associated to the killed process,
        # which are updated below using the FV estimator.
        state_value_killed_process = 0.0
        action_value_killed_process = 0.0
        # Contributions to the state and action values for the killed process come from ALL states of interest, therefore we iterate on them
        for x in self.states_of_interest:
            # Align (in time) the two functions to multiply in the integrand, P(T>t; state, action) and Phi(t, x),
            # so that the same t is associated to each of the two function values being multiplied
            # (recall that x is the state of interest being considered at the current loop, while `state` and `action`
            # are the start state and action w.r.t. which all survival times giving rise to the estimate of P(T>t; state, action)
            # have been measured.
            # Note that this calculation of the integrand assumes that the value of Phi(t,x) does NOT depend on the start state-action
            # associated to the survival function P(T>t; state, action). This assumption is reasonable because we assume that the
            # killed process reaches quasi-stationarity quite fast. If this is deemed as a too strong assumption, we could
            # wait for a burning period before considering contributions to this integral, but I don't think this is really
            # necessary because the contribution from Phi(t,x) for small t is usually quite small (as the rare event is not yet too often observed).

            # Contribution to the state value
            df_phi_proba_surv_for_state = merge_proba_survival_and_phi(df_proba_surv_for_state, self.dict_phi[x], t_origin=self.burnin_time)
            state_value_killed_process += compute_fv_integral(df_phi_proba_surv_for_state, reward=self.env.getTerminalReward(x), interval_size=1/uniform_jump_rate, discount_factor=self.gamma)

            # Contribution to the action value
            df_phi_proba_surv_for_state_action = merge_proba_survival_and_phi(df_proba_surv_for_state_action, self.dict_phi[x], t_origin=self.burnin_time)
            action_value_killed_process += compute_fv_integral(df_phi_proba_surv_for_state_action, reward=self.env.getTerminalReward(x), interval_size=1/uniform_jump_rate, discount_factor=self.gamma)

            ind_gt0 = (df_phi_proba_surv_for_state['P(T>t)'] > 0.0) & (df_phi_proba_surv_for_state['Phi'] > 0.0) & (df_phi_proba_surv_for_state['dt'] > 0.0)
            if np.sum(ind_gt0) > 0:
                print(f"[debug-compute_fv_deltas] s={state}, a={action}:")
                print(f"[debug-compute_fv_deltas] P(T>t; s):\n{df_proba_surv_for_state}")
                print(f"[debug-compute_fv_deltas] Phi:\n{self.dict_phi[x]}")
                print(f"[debug-compute_fv_deltas] f(t,x={x}) = P(T>t; s) * Phi(t,x):\n{df_phi_proba_surv_for_state.loc[ind_gt0, :]}")
                print("[debug-compute_fv_deltas] Integral(f(t,x={})) = {}".format(x, state_value_killed_process))
                print(f"[debug-compute_fv_deltas] alpha[s={state}] = {self._alphas[state]}")
                print("")

        #-- Compute the delta values to be used for the state value and action value functions update
        # The following discount has only an effect when self.gamma < 1.
        # And note that it only makes sense for discrete-time Markov processes, as the exponent is `t_surv - 1`,
        # (i.e. the `-1` correction tells us that t_surv needs to be discrete).
        discount_factor = self.gamma**(t_surv - 1)
        #print(f"[debug] Discount factor applied to the bootstrapping value of the absorption state s={state_absorption}: {discount_factor} (t_surv={t_surv})\n")
        value_at_absorption_state = discount_factor * self.V.getValue(state_absorption)

        # Initialize the delta's to compute to 0
        # The actual values are only computed when we receive a signal from the rare rewards outside A.
        delta_V = delta_Q = 0.0
        if state_value_killed_process != 0:
            delta_V = state_value_killed_process  + value_at_absorption_state - self.V.getValue(state)
        if action_value_killed_process != 0:
            delta_Q = action_value_killed_process + value_at_absorption_state - self.Q.getValue(state, action)

        return delta_V, delta_Q, state_value_killed_process, action_value_killed_process

    def _update_absorption_times(self, t_absorption: int):
        """
        Updates the list of observed absorption times and the dictionary containing the index,
        for each state of interest x and for each observed absorption time,
        that signals the latest entry in dict_phi[x] whose measurement time occurs before the respective
        absorption time (where "respective" means: "stored at the same index position (of the attribute
        storing all observed absorption times) as the index position at which the information in the
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
                # The new entry will have t_absorption - 1 which is the last possible discrete-time value
                # that can be part of the previous inter-absorption interval.
                self.dict_phi[x] = pd.concat([self.dict_phi[x],
                                              pd.DataFrame({'t': [t_absorption - self.TIME_RESOLUTION], 'Phi': [self.dict_phi[x]['Phi'].iloc[-1]]})],
                                             axis=0)
            # Store the last index in Phi(x) that contributes to the latest/current inter-absorption interval
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
                # because at t(i) is when Phi(x) changes value (following the chnage of a step function, which is what Phi(x) is, a step function)
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
                                              pd.DataFrame({'t': [t_absorption], 'Phi': [phi_sum_latest_interval_x]})],
                                             axis=0)

    def _update_integral(self):
        """
        Updates the FV integral with the difference in the area below the product P(T>t) * Phi(t,x),
        for each x among the states of interest.

        This update assumes that all survival times contributing to the P(T>t) estimator occur in increasing order.
        """
        # Number of absorption times observed so far
        n = len(self.absorption_times) - 1  # -1 because the first absorption time in the list is dummy, and is set to 0
        for x in self.states_of_interest:
            first_n_minus1_phi_contributions = self.dict_phi_sum[x]['Phi'].iloc[1:-1]   # We start at index 1 because index 0 contains the dummy value 0, which does not really correspond to a phi contribution to the integral.
            contribution_up_to_n_minus_1 = 1/(n*(n-1)) * np.sum( np.arange(n-1) * first_n_minus1_phi_contributions ) if n > 1 else 0.0
            contribution_from_interval_n = 1/n * self.dict_phi_sum[x]['Phi'].iloc[-1]
            self.dict_integral[x] += contribution_up_to_n_minus_1 + contribution_from_interval_n

    def getNumParticles(self):
        return self.N

    def getNumTimeStepsForExpectation(self):
        return self.T

    def getAbsorptionSet(self):
        return self.absorption_set

    def getActivationSet(self):
        return self.activation_set

    def getProbasStationaryStartStateAbsorption(self):
        return self.probas_stationary_start_state_absorption

    def getProbasStationaryStartStateActivation(self):
        return self.probas_stationary_start_state_activation
