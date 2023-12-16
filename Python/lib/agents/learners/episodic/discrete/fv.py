# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:04:22 2023

@author: Daniel Mastropietro
@description: Fleming-Viot learners on discrete-time, discrete-state, discrete-action environments.
"""

from Python.lib.agents.learners import AlphaUpdateType, LearningCriterion, ResetMethod
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambda, LeaTDLambdaAdaptive


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
    """

    def __init__(self, env, N: int, T: int, absorption_set: set, activation_set: set,
                 probas_stationary_start_state_absorption: dict=None,
                 probas_stationary_start_state_activation: dict=None,
                 criterion=LearningCriterion.AVERAGE,
                 alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
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
        # Dictionary that stores the optional stationary probability of the start states of the single chain simulation
        # and of the N-particle Fleming-Viot process.
        # If None, uniformly random distributions are used.
        self.probas_stationary_start_state_absorption = probas_stationary_start_state_absorption
        self.probas_stationary_start_state_activation = probas_stationary_start_state_activation

        # Note: The average reward that is estimated by the FV learner is already stored in the GenericLearner class
        # from which the Learner class inherits from which the LeaTDLambda class inherits (which is the super class of this class)

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
