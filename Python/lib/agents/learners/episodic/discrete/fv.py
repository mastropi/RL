# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:04:22 2023

@author: Daniel Mastropietro
@description: Fleming-Viot learners on discrete-time, discrete-state, discrete-action environments.
"""

import numpy as np
import pandas as pd

from Python.lib.agents.learners import AlphaUpdateType, LearningCriterion, ResetMethod
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambda, LeaTDLambdaAdaptive
from Python.lib.estimators.fv import initialize_phi, update_phi


class LeaFV(LeaTDLambda):
    """
    Fleming-Viot learner on a discrete environment defined with the DiscreteEnv class of openAI's gym module based on
    learning episodes.

    Learning happens with the following characteristics:
    - The average reward criterion is used (as opposed to discounted criterion). This implies that the discount factor
    gamma is always 1, the episodic task is converted to a continuing task, and the learned value functions are
    the *differential* value functions (i.e. where their estimator, the return G(t), is corrected by an estimate of
    the long-run average reward).
    - The value functions (e.g. V(s)) are learned through the return G(t) estimated with TD(lambda).
    - The average reward is estimated using the Fleming-Viot estimator (of the stationary probability distribution of states).

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
        Set of absorption states, out of which the start state for the single Markov chain simulation
        used to estimate the expected reabsorption cycle time is selected.

    activation_set: set
        Set of activation states, out of which the start state for the N-particle FV process are selected.

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
                 TIME_RESOLUTION=1E-9,
                 debug=False):
        super().__init__(env, criterion=criterion, alpha=alpha,  gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha,
                         alpha_update_type=alpha_update_type, adjust_alpha_by_episode=adjust_alpha_by_episode,
                         alpha_min=alpha_min,
                         store_history_over_all_episodes=True,
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
        # as many indices as the length of self.absorption_times)
        # Ex:
        # self.absorption_times = [0, 3, 8, 13]     (integer-valued!)
        # self.dict_phi[x=9] = {'t': [0, 2, 4, 5, 7, 12], 'Phi': [0, 0.1, 0.0, 0.1, 0.2, 0.2]}
        # self.dict_last_indices_phi_prior_to_absorption_times[x=9] = [0, 1, 4, 5]
        self.dict_last_indices_phi_prior_to_absorption_times = None
        # Phi estimators and their cumulative sum (which is used to update the FV integral when a new absorption is observed)
        self.dict_phi = None
        self.dict_phi_sum = None
        # FV integral value for each state of interest x (updated every time a new absorption is observed)
        self.dict_integral = None

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
            self.dict_phi[x] = pd.DataFrame([[0, 0.0]], columns=['t', 'Phi'])
            self.dict_phi_sum[x] = pd.DataFrame([[0, 0.0]], columns=['t', 'Phi'])
            self.dict_integral[x] = 0.0

    def _update_absorption_times(self, t_absorption: int):
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
            # Store the last index in Phi(x) that contributes to the previous inter-absorption interval
            self.dict_last_indices_phi_prior_to_absorption_times[x] += [ self.dict_phi[x].shape[0] - 1 ]

        # Store the current absorption time
        self.absorption_times += [t_absorption]

    def _update_phi(self, t, state, next_state):
        "Updates the empirical mean phi(t, x) for x in intersection({state, state_next}, states_of_interest)"
        self.dict_phi = update_phi(self.env, self.N, t, self.dict_phi, state, next_state)

    def _update_phi_contribution(self, t_absorption: int):
        "Updates the Phi contribution to the integral coming from the latest interval of events between the previous and the current absorption time"
        assert len(self.absorption_times) > 1, "There are at least 2 absorption times stored in the list of absorption times (meaning there has been at least one observed absorption time)"
        t_absorption_prev = self.absorption_times[-2]
        for x in self.states_of_interest:
            assert self.dict_phi[x].shape[0] > 0, f"The Phi data frame has been initialized for state = {x}"
            assert len(self.dict_last_indices_phi_prior_to_absorption_times[x]) > 1, "At least 2 intervals of events have been observed before the current absorption time"

            phi_sum_last_interval_x = 0.0    # "last" refers to the currently evaluated interval of events observed between the previous absorption time and the current absorption time
            idx_first_phi_x_measurement_in_last_events_interval = self.dict_last_indices_phi_prior_to_absorption_times[x][-2] + 1
            idx_last_phi_x_measurement_in_last_events_interval = self.dict_last_indices_phi_prior_to_absorption_times[x][-1]
            assert idx_first_phi_x_measurement_in_last_events_interval > 0
            for idx in range(idx_first_phi_x_measurement_in_last_events_interval, idx_last_phi_x_measurement_in_last_events_interval + 1):
                t1 = self.dict_phi[x]['t'].iloc[idx-1] if idx > idx_first_phi_x_measurement_in_last_events_interval else t_absorption_prev
                t2 = self.dict_phi[x]['t'].iloc[idx]
                _deltat = t2 - t1
                assert _deltat >= 0, f"t1={t1}, t2={t2}, deltat={_deltat}"
                # The value of Phi(x) contributing to each interval deltat = t(i) - t(i-1) is the value at t(i-1), NOT the value at t(i)
                # because at t(i) is when Phi(x) changes value (following the chnage of a step function, which is what Phi(x) is, a step function)
                _phi = self.dict_phi[x]['Phi'].iloc[idx-1]
                #print(f"delta={_deltat}, Phi={_phi}")
                phi_sum_last_interval_x += _phi * _deltat
            # Contribution to the last micro-interval, [t(last), t_absorption], where t(last) is the time at the last change in Phi recorded at Phi(x)'s row index `idx_last_phi_x_measurement_in_last_events_interval`
            _deltat = t_absorption - self.dict_phi[x]['t'].iloc[idx]
            _phi = self.dict_phi[x]['Phi'].iloc[idx]
            #print(f"delta={_deltat}, Phi={_phi}")
            phi_sum_last_interval_x += _phi * _deltat

            # Store this value in the dictionary of Phi sums
            self.dict_phi_sum[x] = pd.concat([self.dict_phi_sum[x],
                                              pd.DataFrame({'t': [t_absorption], 'Phi': [phi_sum_last_interval_x]})],
                                             axis=0)

    def _update_integral(self):
        "Updates the FV integral with the difference in the area below the product P(T>=t) * Phi(t,x), for each x among the states of interest"
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
