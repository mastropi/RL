# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""

from . import Learner


class LeaPolicyGradient(Learner):
    """
    Policy gradient learner using step size `alpha` and `learnerV` as learner of the state value function

    Arguments:
    env: gym.Env
        The environment where the learning takes place.

    policy: Parameterized policy
        Parameterized policy that is used to learn the parameter theta of the policy.

    learnerV: Learner
        Learner of the state value function.

    alpha: float
        Learning rate.
    """

    def __init__(self, env, policy, learnerV, alpha=0.1,
                 adjust_alpha=False,
                 alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_min)
        self.debug = debug

        self.policy = policy
        self.learnerV = learnerV

        self.dict_state_counts = dict()

    def reset_state_counts(self):
        "Resets the counts of the states visited during the learning process"
        self.dict_state_counts = dict()

    def reset_value_functions(self):
        self.policy.reset()
        self.learnerV.reset()

    def learn(self, t):
        """
        Learns the policy by updating the theta parameter

        Arguments:
        t: float
            Discrete time at which learning takes place.
        """
        theta = self.policy.getThetaParameter()
        state = self.env.getState()
        action = self.env.getLastAction()

        # Observed return
        G = self.learnerV.getV()

        # (2021/10/20) No delta is used for now because we are not using a baseline nor TD(lambda) learner
        #delta = self.learnerV.getVEstimated().getValue(state) - self.learnerV.getV().getValue(state)

        theta += self.alpha * G * self.policy.getGradientLog(action)
        self.policy.setThetaParameter(theta)

    #----- GETTERS -----#
    def getPolicy(self):
        return self.policy

    def getLearnerV(self):
        return self.learnerV
