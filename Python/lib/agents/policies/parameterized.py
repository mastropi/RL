# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropietro
@description: Definition of policy for queues in terms of acceptance of new job of a given class
"""

import numpy as np

import Python.lib.queues as queues
from Python.lib.environments.queues import Actions, EnvQueueSingleBufferWithJobClasses, rewardOnJobClassAcceptance
from Python.lib.agents.policies import GenericParameterizedPolicyTwoActions
from Python.lib.utils.basic import is_scalar


class PolQueueTwoActionsLogit(GenericParameterizedPolicyTwoActions):
    """
    Randomized policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a logit function around parameter theta where the probability of any of the action
    is 0.5.

    Specifically, the probability of acceptance is defined as 1 / (1 + exp( beta*(s - theta) ))
    where beta is a parameter passed to the constructor that controls how steep the transition from probability 1
    to 0 is.

    Arguments:
    env: environment
        Environment representing a queue system.

    theta: float
        Initial value for the reference parameter where the policy is equal to 0.5.

    beta: positive float
        Parameter of the logit that defines the steepness of the transition from probability 1 to 0.
    """

    def __init__(self, env: EnvQueueSingleBufferWithJobClasses, theta: float, beta :float):
        if not is_scalar(theta):
            raise ValueError("The theta parameter must be scalar ({})".format(theta))
        super().__init__(env, theta)

        if self.env.getCapacity() < np.Inf: # When capacity is Inf, theta is also allowed to be Inf
            assert theta < self.env.getCapacity(), "The value of theta is smaller than the queue capacity"

        self.beta = beta

    def getBeta(self):
        return self.beta

    def getGradient(self, action, state):
        """
        Returns the policy gradient at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.

        state: Environment dependent
            State of the environment at which the policy gradient is evaluated.
            The queue's buffer size is computed from the state using the getBufferSizeFromState() method
            defined in the environment object.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        sign = (action.value - 0.5) * 2

        if self.getPolicyForAction(action, state) != 0.0:
            # The following expression is only well defined when the probability for the action is not 0.0
            # But if it is 0.0, the gradient is actually 0.0 because the value of the policy does not change
            # when theta increases infinitesimally.
            gradient = sign * self.beta * (1 - self.getPolicyForAction(action, state))
        else:
            gradient = 0.0

        return gradient

    def getPolicyForAction(self, action, state, buffer_size=None):
        """
        Returns the value of the policy for the given action when the environment is at the given state (or buffer size)

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: Environment dependent
            State of the environment at which the policy is evaluated.
            The queue's buffer size is computed from the state using the getBufferSizeFromState() method
            defined in the environment object.

        buffer_size: (opt) int
            Buffer size triggering the action.
            default: None, in which case the buffer size is computed from the given state.

        Return: float
        Value of the policy for the given action given the environment's state.
        """
        if state is None and buffer_size is None:
            raise ValueError("Either 'state' or 'buffer_size' must be not None")

        if state is not None:
            buffer_size = self.env.getBufferSizeFromState(state)

        policy_accept = 1 / (1 + np.exp( self.beta * (buffer_size - self.theta) ))

        if action == Actions.ACCEPT:
            policy_value_for_action = policy_accept
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - policy_accept
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action


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
        Initial value for the threshold parameter of the policy.
    """

    def __init__(self, env: EnvQueueSingleBufferWithJobClasses, theta: float):
        if not is_scalar(theta):
            raise ValueError("The theta parameter must be scalar ({})".format(theta))
        super().__init__(env, theta)

        if self.env.getCapacity() < np.Inf: # When capacity is Inf, theta is also allowed to be Inf
            assert theta < self.env.getCapacity(), "The value of theta is smaller than the queue capacity"

    def getGradient(self, action, state):
        """
        Returns the policy gradient (with respect to the theta parameter)
        at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.

        state: Environment dependent
            State of the environment at which the policy gradient is evaluated.
            The queue's buffer size is computed from the state using the getBufferSizeFromState() method
            defined in the environment object.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        buffer_size = self.env.getBufferSizeFromState(state)

        is_buffer_size_in_linear_piece = float( self.theta < buffer_size < self.theta + 1 )
        # Recall that the gradient is w.r.t. theta, NOT w.r.t. to the buffer size
        # which is how we normally plot the policy and against which the derivative has the opposite sign
        # compared to the sign of the derivative w.r.t. theta.
        if action == Actions.ACCEPT:
            slope = is_buffer_size_in_linear_piece
        elif action == Actions.REJECT:
            slope = -is_buffer_size_in_linear_piece
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return slope

    def getPolicyForAction(self, action, state, buffer_size=None):
        """
        Returns the value of the policy for the given action when the queue environment is at the given state or
        buffer size, i.e. it returns the *probability* that the given action is taken when the system is at the given state.

        The calculation of the policy is based on either the `state` or the `buffer_size`, whichever is not none.
        If both are given, the state has precedence and the buffer size is computed from it.

        In all cases, since the policy's parameter is univariate, the value of the policy is a function of the buffer size
        (either given or computed from the state), which is compared to the current value of the theta parameter of the policy.

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: Environment dependent
            State of the environment at which the policy is evaluated.
            If not None, the queue's buffer size is computed from the state using the getBufferSizeFromState() method
            defined in the environment object.
            If None, the buffer_size parameter must not be None.

        buffer_size: (opt) int
            Buffer size at which the policy of the given action is requested.
            default: None, in which case the buffer size is computed from the given state.

        Return: float
        Value of the policy for the given action given the environment's state.
        """
        if state is None and buffer_size is None:
            raise ValueError("Either 'state' or 'buffer_size' must be not None")

        if state is not None:
            buffer_size = self.env.getBufferSizeFromState(state)

        if buffer_size <= self.theta[0]:
            policy_accept = 1.0
        elif buffer_size >= self.theta[0] + 1:
            policy_accept = 0.0
        else:
            # theta < s < theta + 1
            # => The acceptance policy is linear in s
            policy_accept = self.theta[0] - buffer_size + 1

        if action == Actions.ACCEPT:
            policy_value_for_action = policy_accept
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - policy_accept
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action

    def getBufferSizeForDeterministicBlocking(self):
        "Returns K, the first integer greater than the theta parameter stored in the object + 1"
        return self.getBufferSizeForDeterministicBlockingFromTheta(self.getThetaParameter()[0])

    def getBufferSizeForDeterministicBlockingFromTheta(self, theta):
        "Returns K, the first integer greater than theta + 1"
        return int( np.ceil( theta + 1 ) )


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
        if len(self.getThetaParameter()) != self.env.getCapacity() + 1:
            raise ValueError("The number of theta parameters given ({}) must be equal to the capacity of the queue plus 1 ({})".format(len(theta), env.getCapacity() + 1))

        # We now define the linear step policy
        # NOTE: if retrieving the policy becomes somehow inefficient for choosing the action
        # (because we are storing the policy for ALL the states)
        # we may want to AVOID the definition of the policy, as, to determine the action to take,
        # we just need to know the job class falling in between the two different deterministic selections of an action.
        K = self.env.getCapacity()
        J = self.env.getNumJobClasses()

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

    def getGradient(self, action, state):
        """
        Returns the policy gradient at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.

        state: tuple
            State of the environment at which the policy gradient is evaluated.
            It should be a duple containing the queue's buffer size and the job class of the latest arriving job.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        buffer_size, job_class = state

        is_job_class_in_linear_piece = self.theta[buffer_size] < job_class < self.theta[buffer_size] + 1
        if action == Actions.ACCEPT:
            slope = is_job_class_in_linear_piece
        elif action == Actions.REJECT:
            slope = -is_job_class_in_linear_piece
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return slope

    def getPolicyForAction(self, action, state):
        """
        Returns the value of the policy for the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: tuple
            State of the environment at which the policy is evaluated.
            It should be a duple containing the queue's buffer size and the job class of the latest arriving job.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        buffer_size, job_class = state

        if action == Actions.ACCEPT:
            policy_value_for_action = self.policy_accept[buffer_size, job_class]
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - self.policy_accept[buffer_size, job_class]
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action

    def getAcceptancePolicy(self):
        return self.policy_accept


if __name__ == "__main__":
    #------------------------- Unit tests -----------------------------#
    #--- Test #1: Single-server, single-class job: stepwise linear policy on buffer size
    print("\n***********\nTest #1: Tests on Stepwise-linear policy on a single server with single threshold parameter theta")
    job_class_rates = [0.7]
    job_rates_by_server = job_class_rates     # Since this is a single server system on a SINGLE class, the job arriving rate to the server is the same as the job class rate
    service_rates = [1]
    nservers = 1
    capacity = np.Inf
    rewards_accept_by_job_class = [0]
    queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
    env = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobClassAcceptance, rewards_accept_by_job_class)

    #-- Thresholds
    # theta <= 0: No new job is accepted
    # theta very close to 0
    theta = -0.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)

    buffer_size = 0
    env.setQueueState(buffer_size)
    state = env.getState()
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.7
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size = 1
    env.setQueueState(buffer_size)
    state = env.getState()
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # theta a little farther away than 0
    theta = -1.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)
    for buffer_size in range(3):
        env.setQueueState(buffer_size)
        state = env.getState()
        print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
        assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
        assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # theta > 0, non-integral
    theta = 5.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env, theta)

    buffer_size_small = 3
    env.setQueueState(buffer_size_small)
    state = env.getState()
    print("\tTesting buffer_size smaller than theta: {}... Policy Accept = {}".format(buffer_size_small, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_large = 8
    env.setQueueState(buffer_size_large)
    state = env.getState()
    print("\tTesting buffer_size = {}, MUCH larger than theta... Policy Accept = {}".format(buffer_size_large, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_border_blocked = int(theta+2)
    env.setQueueState(buffer_size_border_blocked)
    state = env.getState()
    print("\tTesting buffer_size = {}, near theta with sure blocking... Policy Accept = {}".format(buffer_size_border_blocked, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_border_random = int(theta+1)
    env.setQueueState(buffer_size_border_random)
    state = env.getState()
    print("\tTesting buffer_size = {}, near theta with randomized blocking... Policy Accept = {}".format(buffer_size_border_random, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert np.isclose(policy.getPolicyForAction(Actions.ACCEPT, state), 0.3)
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    #--- Test #2: Single-server, multi-class job: stepwise linear policy
    print("\n***********\nTest #2: Tests on Stepwise-linear policy on a single server with one threshold parameter theta for each job class")
    job_class_rates = [0.7, 0.3, 0.4, 0.9]  # Note: This rates are currently NOT used in the test. The list just defines the number of possible job classes
    job_rates_by_server = [0.7]             # This value is not used at all, it's just needed to create the queue system below
    service_rates = [1]                     # This value is not used in the test but needed to create the queue system below
    nservers = 1
    capacity = 6
    rewards_accept_by_job_class = [1, 0.8, 0.3, 0.2]
    queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
    env = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobClassAcceptance, rewards_accept_by_job_class)

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
    env.setState((buffer_size, job_class))
    state = env.getState()
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # 1.2: In this setup, jobs with lowest priority are accepted with non-zero probability lower than 1
    # regardless of the buffer size
    job_class = 2
    env.setState((3, job_class))
    state = env.getState()
    print("Prob(action=1/s={},theta={:.1f}) for a job class {} (falling between theta and theta+1): {}" \
          .format(env.getJobClass(), env.getBufferSize(), policy.theta[job_class], policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert 0 < policy.getPolicyForAction(Actions.ACCEPT, state) and policy.getPolicyForAction(Actions.ACCEPT, state) < 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)
    assert np.abs(policy.getPolicyForAction(Actions.ACCEPT, state) - 0.30) < 1E-6

    # 1.3: Any job is rejected when the buffer is full
    env.setState((env.queue.getCapacity(), 0))
    state = env.getState()
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
