# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:04:22 2023

@author: Daniel Mastropietro
@description: Fleming-Viot learners on discrete-time, discrete-state, discrete-action environments.
"""

from Python.lib.agents.learners import AlphaUpdateType, LearningCriterion, ResetMethod
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambda


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
    """

    def __init__(self, env, N, T, absorption_set, activation_set, probas_stationary_start_state: dict=None,
                 alpha=0.1, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 debug=False):
        super().__init__(env, criterion=LearningCriterion.AVERAGE, alpha=alpha,  gamma=1.0, lmbda=lmbda, adjust_alpha=adjust_alpha,
                         alpha_update_type=alpha_update_type, adjust_alpha_by_episode=adjust_alpha_by_episode,
                         alpha_min=alpha_min,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed, debug=debug)

        # Fleming-Viot simulation parameters
        self.N = N
        self.T = T
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set ({}).".format(type(absorption_set)))
        if not isinstance(activation_set, set):
            raise ValueError("Parameter `activation_set` must be a set ({}).".format(type(activation_set)))
        self.absorption_set = absorption_set
        self.activation_set = activation_set
        # Dictionary that stores the optional stationary probability of the start states of the Fleming-Viot simulation
        # run on N particles.
        # If None, a uniformly random selection of the start state is done.
        self.probas_stationary_start_state = probas_stationary_start_state

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

    def getProbasStationaryStartState(self):
        return self.probas_stationary_start_state
