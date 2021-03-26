# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropetro
@description: Definition of policy for queues in terms of acceptance of new job of a given class
"""

import numpy as np

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses

class PolQueueRandomizedLinearStep:
    """
    Randomized policy where the transition from acceptance to rejection is
    a stepwise linear function of the job class.
    
    It is assumed that the possible actions are 0 (reject) or 1 (accept).
    
    The probability for action 1 is:
    - 1.0 if job_class <= threshold[k]
    - 0.0 if job_class >= threshold[k} = 1
    - threshold[k] - job_class + 1 o.w. 
    where k is the current buffer size of the queue.

    Note that if threshold[k] is integer, the policy is deterministic.
    
    Ref: Massaro et al. "Optimal Trunk-Reservation by Policy Learning", pag. 3
    """

    def __init__(self, env :EnvQueueSingleBufferWithJobClasses, thresholds: list):
        if len(thresholds) != env.getCapacity() + 1:
            raise ValueError("The number of threshold parameters given ({}) must be equal to the capacity of the queue plus 1 ({})".format(len(thresholds), env.getCapacity()+1))

        self.env = env
        self.thresholds = thresholds

        # We now define the step policy, but this is just for INFORMATIONAL PURPOSES ONLY
        # it is NOT explicitly used when choosing an action, because it is not so efficient,
        # i.e. we don't need to store all the information we are storing now in the policy,
        # we just need to know the job class falling in between the deterministic selection of an action.
        K = env.getCapacity()
        J = env.getNumJobClasses()

        self.policy = {}
        for a in env.getActions()[1:]:
            # Policy = 0.0 for j >= theta[k] + 1
            self.policy[a] = np.zeros((K+1, J), dtype=float)
            for k in range(K+1):
                #print("k={} of K={}".format(k, K))
                for j in range( np.min( [int(thresholds[k])+1, J]) ):
                    #print("1: j={}".format(j))
                    # Policy = 1.0 for j <= theta[k]
                    self.policy[a][k,j] = 1.0
                for j in range( np.min( [int(thresholds[k])+1, J-1] ), np.min( [int(thresholds[k])+2, J-1] )):
                    #print("2: j={}".format(j))
                    # Policy = theta[k] - j + 1 for theta[k] < j < theta[k] + 1
                    self.policy[a][k,j] = thresholds[k] - j + 1
                #print(self.policy[a][k])

    def choose_action(self):
        """
        Choose an action, either 0 (reject) or 1 (accept) based on the policy
        evaluated on the current state of the environment.
        """ 
        prob_action_1 = self.getProbability(1)
        if prob_action_1 == 1.0:
            return 1
        elif prob_action_1 == 0.0:
            return 0
        else:
            # Choose action 1 with prob_action_1 probability
            return self.env.np_random() < prob_action_1

    def getGradient(self, action):
        "Returns the policy gradient at the given action and current state of the environment"
        buffer_size, job_class = self.env.getState()

        is_job_class_in_linear_piece = not (job_class <= self.thresholds[buffer_size] or job_class >= self.thresholds[buffer_size] + 1)
        if action == 1:
            return is_job_class_in_linear_piece
        else:
            return -is_job_class_in_linear_piece

    def getGradientLog(self, action):
        "Returns the gradient of the log policy at the given action and current state of the environment"
        policy_value = self.getProbability(action)
        if 0 < policy_value and policy_value < 1:
            return self.getGradient(action) / policy_value
        else:
            return 0.0

    def getProbability(self, action):
        """
        Returns the value of the policy for the given action at the current state of the environment.
        
        It is assumed that `action` is either 0 or 1.
        """
        buffer_size, job_class = self.env.getState()

        prob_action_1 = None
        if buffer_size == self.env.getCapacity():
            prob_action_1 = 0.0
        else:
            #print(self.thresholds)
            if job_class <= self.thresholds[buffer_size]:
                prob_action_1 = 1.0
            elif job_class >= self.thresholds[buffer_size] + 1:
                prob_action_1 = 0.0
            else:
                prob_action_1 = self.thresholds[buffer_size] - job_class + 1

        if action == 1:
            return prob_action_1
        else:
            return 1 - prob_action_1

    def getPolicy(self):
        return self.policy


if __name__ == "__main__":
    #------------------------- Unit tests -----------------------------#
    #--- Test #1: Stepwise linear policy
    env = EnvQueueSingleBufferWithJobClasses(6, 4, [1, 0.8, 0.3, 0.2])
    theta = list( reversed( [0, 1, 1.3, 1.3, 2.8, 2.9, 2.9] ))   # Must of length capacity+1 (the threshold values are given from top (higher buffer size) to bottom (lower buffer size, and then reversed) 
    print("theta: {}".format(theta))
    policy = PolQueueRandomizedLinearStep(env, theta)

    # Highest priority job is always accepted
    # regardless of the buffer size (unless the buffer is full)
    env.setJobClass(0)
    assert policy.getProbability(1) == 1.0
    assert policy.getProbability(0) == 1 - policy.getProbability(1)

    # In this setup, jobs with lowest priority are accepted with non-zero probability lower than 1 
    # regardless of the buffer size
    env.setJobClass(3)
    print("Prob(action=1) for a job class with randomized policy: {}".format(policy.getProbability(1)))
    assert 0 < policy.getProbability(1) and policy.getProbability(1) < 1.0
    assert policy.getProbability(0) == 1 - policy.getProbability(1)

    # Any job is rejected when the buffer is full
    env.setBufferSize(env.capacity)
    env.setJobClass(0)
    assert policy.getProbability(1) == 0.0
