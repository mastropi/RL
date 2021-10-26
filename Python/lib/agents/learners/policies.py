# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""
import numpy as np

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

        self.baseline = 0.0                 # Baseline used in Delta(t) used in the theta update
        self.delta = 0.0                    # Delta(t) = G(t)

        # Attributes that store the history of the learning process
        self.actions = []
        self.states = []
        self.gradients = []
        self.log_gradients = []

        self.dict_state_counts = dict()

    def reset_supporting_attributes(self):
        "Resets the counts of the states visited during the learning process"
        self.baseline = 0.0
        self.delta = 0.0
        self.gradients = []
        self.log_gradients = []
        self.dict_state_counts = dict()

    def reset_value_functions(self):
        self.policy.reset()
        self.learnerV.reset()

    def record_gradient(self, state, action, gradient):
        self.gradients += [gradient]

        policy_value = self.policy.getPolicyForAction(action, state)
        log_gradient = gradient / policy_value if policy_value != 0.0 else 0.0 if gradient == 0.0 else np.nan
        self.log_gradients += [log_gradient]

    def learn(self, t, state, action):
        """
        Learns the policy by updating the theta parameter based on the state and the action taken on that state

        Arguments:
        t: float
            Discrete time at which learning takes place.

        state: Environment dependent
            State of the environment.
            This is the state at which the policy gradient is to be evaluated when learning.

        action: Environment dependent
            Last action taken by the agent on the environment among all possible actions defined in the environment's
            action space.
            This is the action at which the policy gradient for the learning step is to be evaluated when learning.
        """
        theta = self.policy.getThetaParameter()

        # Observed return for every t
        # This should be the return already adjusted by the baseline
        G = self.learnerV.getV()

        # Usual definition of delta: estimated state value V with the new observed reward - previous estimated state value
        #self.delta = self.learnerV.getVEstimated().getValue(state) - self.learnerV.getV().getValue(state)
        # Delta is just equal to the return (which may already be affected by the baseline --see Trunk Reservation paper page 4)
        #self.delta = G
        # Delta as the return minus a baseline chosen as the state value estimated in the previous learning step
        # (which is assumed to have been used as STARTING value of V when resetting the learner)
        #baseline = self.learnerV.getVStart()
        self.baseline = np.mean( self.learnerV.getVHist() )
        self.delta = G - self.baseline

        if action is not None:
            ## Note: the action may be None if for instance the environment experienced a completed service
            ## in which case there is no action to take... (in terms of Accept or Reject)

            # When using the latest log-gradient observed in the queue simulation
            # Note that we bound the delta theta to (-2, 2) to avoid large changes!
            #theta += self.alpha * delta * self.policy.getGradientLog(action, state)
            theta += np.max([ np.min([2.0, self.alpha * self.delta * self.policy.getGradientLog(action, state)]), -2.0 ])

            # When using the average log-gradient observed in the queue simulation
            #average_log_gradient = np.mean( self.getLogGradients() )
            #theta += self.alpha * G * average_log_gradient
        self.policy.setThetaParameter(theta)
        self.policy.update_thetas(theta)

    #----- GETTERS -----#
    def getPolicy(self):
        return self.policy

    def getLearnerV(self):
        return self.learnerV

    def getGradients(self):
        return self.gradients

    def getLogGradients(self):
        return self.log_gradients

    def getBaseline(self):
        return self.baseline

    def getDelta(self):
        return self.delta
