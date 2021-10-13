# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropetro
@description: Definition of policy for queues in terms of acceptance of new job of a given class
"""

import numpy as np

import Python.lib.queues as queues
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, \
    deprecated2_EnvQueueSingleBufferWithJobClasses, rewardOnJobClassAcceptance
from agents.policies import GenericParameterizedPolicyTwoActions


class PolQueueTwoActionsLinearStep(GenericParameterizedPolicyTwoActions):
    """
    Randomized policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a stepwise linear function that transitions from deterministic acceptance to deterministic rejection.
    The location of the linear transition from acceptance to rejection happens at only one buffer size and is defined
    by the threshold parameter `theta`.

    Specifically, the probability of acceptance is defined as:
    - 1.0 if s <= theta
    - 0.0 if s >= theta + 1
    - theta - s + 1.0, o.w.
    where s is the current buffer size of the queue.

    Note that if theta is integer, the policy is deterministic for all buffer sizes.

    Ref: Massaro et al. (2019) "Optimal Trunk-Reservation by Policy Learning", pag. 3, from where we adapt
    the parameterized policy defined for multiple-class jobs.

    Arguments:
    env: environment
        Environment representing a queue system.

    theta: float
        Threshold parameter.
    """

    def __init__(self, env: EnvQueueSingleBufferWithJobClasses, theta: float):
        if not isinstance(theta, float):
            raise ValueError("The theta parameter must be a float number ({})".format(theta))
        super().__init__(env, theta)

        # We now define the linear step policy
        # NOTE: if retrieving the policy becomes somehow inefficient for choosing the action
        # (because we are storing the policy for ALL the states)
        # we may want to AVOID the definition of the policy, as, to determine the action to take,
        # we just need to know the buffer size falling in between the two different deterministic selections of an action.
        K = env.getCapacity()
        assert K == np.Inf, "The queue has infinite (fixed) capacity, since parameter theta defining its capacity can take ANY real value."

    def getGradient(self, action):
        "Returns the policy gradient at the given action and current state of the environment"
        buffer_size = self.env.getBufferSize()

        is_buffer_size_in_linear_piece = self.theta < buffer_size < self.theta + 1
        if action == 1:
            return is_buffer_size_in_linear_piece
        else:
            return -is_buffer_size_in_linear_piece

    def getPolicyForAction(self, action):
        "Returns the value of the policy for the given action at the current state of the environment"
        buffer_size = self.env.getBufferSize()

        if buffer_size <= self.theta[0]:
            policy_accept = 1.0
        elif buffer_size >= self.theta[0] + 1:
            policy_accept = 0.0
        else:
            # theta < s < theta + 1
            # => The acceptance policy is linear in s
            policy_accept = self.theta[0] - buffer_size + 1

        if action == 1:
            return policy_accept
        else:
            return 1 - policy_accept


class PolQueueTwoActionsLinearStepOnJobClasses(GenericParameterizedPolicyTwoActions):
    """
    Randomized policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a stepwise linear function that transitions from deterministic acceptance to deterministic rejection.
    The location of the linear transition from acceptance to rejection happens at only one job class and is defined
    by the threshold parameter `theta`.

    Job classes can be thought of as job priorities.

    Specifically, the probability of acceptance is defined as:
    - 1.0 if job_class <= theta[k]
    - 0.0 if job_class >= theta[k] + 1
    - theta[k] - job_class + 1.0, o.w.
    where k is the current buffer size of the queue.

    Note that if theta[k] is integer, the policy is deterministic for all job classes.
    
    Ref: Massaro et al. (2019) "Optimal Trunk-Reservation by Policy Learning", pag. 3

    Arguments:
    env: environment
        Environment representing a queue system that accepts multiple job classes.

    theta: list
        Threshold parameter. Its length is equal to the capacity of the queue system.
    """

    def __init__(self, env :EnvQueueSingleBufferWithJobClasses, theta: list):
        super().__init__(env, theta)
        if len(theta) != env.getCapacity() + 1:
            raise ValueError("The number of theta parameters given ({}) must be equal to the capacity of the queue plus 1 ({})".format(len(theta), env.getCapacity() + 1))

        # We now define the linear step policy
        # NOTE: if retrieving the policy becomes somehow inefficient for choosing the action
        # (because we are storing the policy for ALL the states)
        # we may want to AVOID the definition of the policy, as, to determine the action to take,
        # we just need to know the job class falling in between the two different deterministic selections of an action.
        K = env.getCapacity()
        J = env.getNumJobClasses()

        # Initialize acceptance policy to 0.0 which applies to job classes j >= theta[k] + 1
        self.policy_accept = np.zeros((K + 1, J), dtype=float)

        # Define acceptance policy with non-zero probability
        # for buffer sizes < K and job classes j <= theta[k]
        for k in range(K+1):
            max_jobclass_with_accept_proba_1 = int( np.min([np.floor(theta[k])  , J-1]) )
            min_jobclass_with_accept_proba_0 = int( np.min([np.floor(theta[k])+1, J-1]) )
            #print("k={} of K={}: max_j_proba_1={}, min_j_proba_0={}".format(k, K, max_jobclass_with_accept_proba_1, min_jobclass_with_accept_proba_0))
            assert max_jobclass_with_accept_proba_1 == min_jobclass_with_accept_proba_0 - 1 if max_jobclass_with_accept_proba_1 < J-1 \
                else max_jobclass_with_accept_proba_1 == min_jobclass_with_accept_proba_0
            for j in range(max_jobclass_with_accept_proba_1 + 1):
                # Policy of accept = 1.0 for j <= theta[k]
                #print("1: j={}".format(j))
                self.policy_accept[k, j] = 1.0
            for j in range(max_jobclass_with_accept_proba_1 + 1, min_jobclass_with_accept_proba_0 + 1):
                # Policy of accept = theta[k] - j + 1 for theta[k] < j < theta[k] + 1
                #print("2: j={}".format(j))
                self.policy_accept[k, j] = theta[k] - j + 1
            print("Acceptance policy as a function of the job class for buffer size k = {} (theta[k]={:.1f}): {}".format(k, theta[k], self.policy_accept[k]))

    def getGradient(self, action):
        "Returns the policy gradient at the given action and current state of the environment"
        buffer_size, job_class = self.env.getState()

        is_job_class_in_linear_piece = self.theta[buffer_size] < job_class < self.theta[buffer_size] + 1
        if action == 1:
            return is_job_class_in_linear_piece
        else:
            return -is_job_class_in_linear_piece

    def getPolicyForAction(self, action):
        "Returns the value of the policy for the given action at the current state of the environment"
        buffer_size, job_class = self.env.getState()

        if action == 1:
            return self.policy_accept[buffer_size, job_class]
        else:
            return 1 - self.policy_accept[buffer_size, job_class]

    def getAcceptPolicy(self):
        return self.policy_accept


if __name__ == "__main__":
    #------------------------- Unit tests -----------------------------#
    #--- Test #1: Stepwise linear policy on buffer size on a single server system
    job_class_rates = [0.7]
    job_rates = job_class_rates     # Since this is a single server system on a SINGLE class, the job arriving rate to the server is the same as the job class rate
    service_rates = [1]
    nservers = 1
    capacity = np.Inf
    rewards_accept_by_job_class = [0]
    queue = queues.QueueMM(job_rates, service_rates, nservers, capacity)
    env = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobClassAcceptance, rewards_accept_by_job_class)

    #-- Thresholds
    # theta <= 0: No new job is accepted
    # theta very close to 0
    theta = -0.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)

    buffer_size = 0
    env.setBufferSize(buffer_size)
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(1)))
    assert policy.getPolicyForAction(1) == 0.7
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    buffer_size = 1
    env.setBufferSize(buffer_size)
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(1)))
    assert policy.getPolicyForAction(1) == 0.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    # theta a little farther away than 0
    theta = -1.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)
    for buffer_size in range(3):
        env.setBufferSize(buffer_size)
        print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(1)))
        assert policy.getPolicyForAction(1) == 0.0
        assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    # theta > 0, non-integral
    theta = 5.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)

    buffer_size_small = 3
    env.setBufferSize(buffer_size_small)
    print("\tTesting buffer_size smaller than theta: {}... Policy Accept = {}".format(buffer_size_small, policy.getPolicyForAction(1)))
    assert policy.getPolicyForAction(1) == 1.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    buffer_size_large = 8
    env.setBufferSize(buffer_size_large)
    print("\tTesting buffer_size = {}, MUCH larger than theta... Policy Accept = {}".format(buffer_size_large, policy.getPolicyForAction(1)))
    assert policy.getPolicyForAction(1) == 0.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    buffer_size_border_blocked = int(theta+2)
    env.setBufferSize(buffer_size_border_blocked)
    print("\tTesting buffer_size = {}, near theta with sure blocking... Policy Accept = {}".format(buffer_size_border_blocked, policy.getPolicyForAction(1)))
    assert policy.getPolicyForAction(1) == 0.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    buffer_size_border_random = int(theta+1)
    env.setBufferSize(buffer_size_border_random)
    print("\tTesting buffer_size = {}, near theta with randomized blocking... Policy Accept = {}".format(buffer_size_border_random, policy.getPolicyForAction(1)))
    assert np.isclose(policy.getPolicyForAction(1), 0.3)
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    #--- Test #2: Stepwise linear policy on multi-class job (on deprecated queue environment)
    capacity = 6
    num_job_classes = 4
    rewards_accept_by_job_class = [1, 0.8, 0.3, 0.2]
    env = deprecated2_EnvQueueSingleBufferWithJobClasses(capacity, num_job_classes, rewards_accept_by_job_class)
    # Thresholds as a function of the buffer size (length = capacity+1)
    # (the threshold values are given from top (higher buffer size) to bottom (lower buffer size) and then reversed)
    # Note the following extreme cases for testing purposes:
    # - the threshold for the largest job class is set to be negative
    # - the threshold for job class 0 is set way larger than the queue's capacity
    theta = list( reversed( [-2.0, 1.0, 1.3, 1.3, 2.8, 2.9, 5.0] ))
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStepOnJobClasses(env, theta)

    # 1.1: Highest priority job is always accepted
    # regardless of the buffer size (unless the buffer is full)
    job_class = 0
    env.setJobClass(job_class)
    assert policy.getPolicyForAction(1) == 1.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)

    # 1.2: In this setup, jobs with lowest priority are accepted with non-zero probability lower than 1
    # regardless of the buffer size
    job_class = 2
    env.setJobClass(job_class)
    env.setBufferSize(3)
    print("Prob(action=1/s={},theta={:.1f}) for a job class {} (falling between theta and theta+1): {}" \
          .format(env.getJobClass(), env.getBufferSize(), policy.theta[job_class], policy.getPolicyForAction(1)))
    assert 0 < policy.getPolicyForAction(1) and policy.getPolicyForAction(1) < 1.0
    assert policy.getPolicyForAction(0) == 1 - policy.getPolicyForAction(1)
    assert np.abs(policy.getPolicyForAction(1) - 0.30) < 1E-6

    # 1.3: Any job is rejected when the buffer is full
    env.setBufferSize(env.capacity)
    env.setJobClass(0)
    assert policy.getPolicyForAction(1) == 0.0
