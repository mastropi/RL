# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropetro
@description: Definition of environments on queues
"""

from warnings import warn
import numpy as np

import gym
from gym import spaces

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy

    runpy.run_path('../../../setup.py')

    from Python.lib.environments import EnvironmentDiscrete
    from Python.lib.queues import GenericQueue, QueueMM
else:
    from . import EnvironmentDiscrete
    from ..queues import GenericQueue, QueueMM

#---------------------------- Reward functions that can be used in different environments -----------------------------#
# Each reward function should depend on 4 inputs:
# - env: the environment  where the reward is generated
# - S(t) the state at which an action is applied
# - A(t) the action applied at state S(t)
# - S(t+1) the next state where the system is at after applying action A(t) on state S(t)
def rewardOnJobClassAcceptance(env, state, action, next_state):
    # The state is assumed to be a tuple (k, i) where k is the buffer size and i is the class of the arriving job
    job_class = state[1]
    return env.getRewardForJobClassAcceptance(job_class)
#---------------------------- Reward functions that can be used in different environments -----------------------------#


class EnvQueueSingleBufferWithJobClasses(gym.Env):
    """
    Queue environment with a single buffer receiving jobs of different classes
    being served by one or multiple servers.

    The possible actions for an agent interacting with the queue are:
    - Accept or reject an incoming job of a given class
    - Choose the server to which the accepted incoming job is assigned

    Arguments:
    queue: QueueMM
        The queue object that governs the dynamics of the environment through the definition of a server system.

    job_class_rates: list
        A list containing the arriving job rates for each valid job class.

    reward_func: function
        Function returning the reward received after taking action A(t) at state S(t) and making the environment
        go to state S(t+1).

    rewards_accept_by_job_class: list
        List of rewards for the acceptance of the different job class whose rate is given in job_class_rates.

    policy_accept: Policy
        Policy object that defines the acceptance policy.
        Note that the policy is used by the agent when interacting with the environment, and it's stored here as part
        of the information that we store about the environment, because not every policy is compatible with an
        environment.

    policy_assign: (opt) list of lists
        Assignment policy defined as a list of assignment probabilities for each job class
        (associated to job rates) to a server in the queue.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        default: None (in which case the assignment probability is uniform over the servers)
    """

    def __init__(self, queue: QueueMM, job_class_rates: list, reward_func, rewards_accept_by_job_class: list, policy_accept=None, policy_assign: list = None):
        #-- Environment
        self.queue = queue
        self.job_class_rates = job_class_rates

        #-- Rewards
        self.reward_func = reward_func
        self.rewards_accept_by_job_class = rewards_accept_by_job_class

        #-- Policies
        # TODO-2021/10/12: We should receive an Agent object instead of just the Policy... The Agent would include the Policy AND the Learner
        self.policy_accept = policy_accept
        # Job-class to server assignment policy
        if policy_assign is None:
            # When no assignment probability is given, it is defined to be uniform over the servers for each job class
            policy_assign = [[1.0 / self.queue.getNServers()] * self.queue.getNServers()] * len(self.job_class_rates)
        self.policy_assign = policy_assign

        # Job-rates by server (assuming jobs are pre-assigned to servers)
        self.job_rates_by_server = self.compute_arrival_rates_by_server()

        # Create temporary variables that are shorter to reference and still easy to understand
        K = self.queue.getCapacity()
        J = len(self.job_class_rates)

        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # Either Reject (0) or Accept (1) an arriving job class
        self.observation_space = spaces.Tuple((spaces.Discrete(K+1), spaces.Discrete(J)))

        self.reset()

    def reset(self):
        self.setState(0, None)

    def step(self, action):
        assert self.getJobClass() is not None, "The job class of the arrived job is defined"

        if not self.action_space.contains(action):
            raise ValueError(
                "Invalid action: {}\nValid actions are: 0, 1, ..., {}".format(action, self.action_space.n - 1))

        if action == 0:
            # Reject the incoming job
            # Set the job class to None because no new job has yet arrived at the moment of taking the action
            self.setState(self.getBufferSize(), None)
            next_state = self.getState()
            reward = 0.0
        elif action == 1:
            # Accept the incoming job, if the queue has capacity
            if self.getBufferSize() == self.queue.getCapacity():
                # Set the job class to None because no new job has yet arrived at the moment of taking the action
                self.setState(self.getBufferSize(), None)
                next_state = self.getState()
                reward = 0.0

                #warn("Action '{}' not taken because the buffer is at full capacity ({})!".format(action,
                #                                                                                 self.getCapacity()))
            else:
                # Current state of the system
                state = self.getState()
                # Change the state of the system
                self.setState(self.getBufferSize()+1, None)
                next_state = self.getState()
                # Reward observed by going from S(t) to S(t+1) via action A(t)
                reward = self.reward_func(self, state, action, next_state)

        return next_state, reward, False, {}

    def compute_arrival_rates_by_server(self):
        """
        Computes the equivalent job arrival rates for each server from the job arrival rates (to the single buffer)
        and the assignment policy.

        This can be used when the job is pre-assigned to a server at the moment of arrival (as opposed to being
        assigned when a server queue is freed.

        Return: list
            The equivalent job arrival rates for each server r, computed as:
            job_arrival_rate[r] = sum_{c over job classes} { job_rate[c] * Pr(assign job c to server r) }            
        """
        R = self.queue.getNServers()
        J = len(self.job_class_rates)  # Number of job classes
        job_rates_by_server = [0]*R
        for r in range(R):
            for c in range(J):
                job_rates_by_server[r] += self.policy_assign[c][r] * self.job_class_rates[c]

        return job_rates_by_server

    # ------ GETTERS ------#
    def getCapacity(self):
        return self.queue.getCapacity()

    def getJobRates(self):
        return self.job_class_rates

    def getJobRatesByServer(self):
        return self.job_rates_by_server

    def getServiceRates(self):
        return self.queue.getDeathRates()

    def getIntensities(self):
        return [b/d for b, d in zip(self.getJobRatesByServer(), self.getServiceRates())]

    def getAssignPolicy(self):
        return self.policy_assign

    def getNumJobClasses(self):
        return len(self.job_class_rates)

    def getNumServers(self):
        return self.queue.getNServers()

    def getState(self):
        return (self.getBufferSize(), self.getJobClass())

    def getBufferSize(self):
        return self.buffer_size

    def getJobClass(self):
        return self.job_class

    def getActions(self):
        return range(self.action_space.n)

    def getRewardsForJobClassAcceptance(self):
        return self.rewards_accept_by_job_class

    def getRewardForJobClassAcceptance(self, job_class: int):
        return self.rewards_accept_by_job_class[job_class]

    # ------ SETTERS ------#
    def setBufferSize(self, buffer_size: int):
        "Sets the buffer size of the queue"
        if int(buffer_size) != buffer_size or buffer_size < 0 or buffer_size > self.getCapacity():
            raise ValueError(
                "The buffer size value is invalid ({}). It must be an integer between {} and {}".format(buffer_size, 0,
                                                                                                        self.getCapacity()))
        self.buffer_size = buffer_size

    def setJobClass(self, job_class: int or None):
        "Sets the job class of a new arriving job. The job class can be None."
        if  not (
                job_class is None or \
                int(job_class) == job_class and 0 <= job_class <= self.getNumJobClasses()
                ):
            raise ValueError(
                "The job class is invalid ({}). It must be an integer between {} and {}".format(job_class, 0,
                                                                                                self.getNumJobClasses() - 1))
        self.job_class = job_class

    def setState(self, buffer_size: int, job_class: int):
        "Sets the state (k, i) where k is the buffer size and i is the arriving job class"
        self.buffer_size = buffer_size
        self.job_class = job_class


if __name__ == "__main__":
    # ------------------------- Unit tests -----------------------------#
    # --- Test #1: Create a queue environment
    print("\nRunning test #3: Queue environment with a GenericQueue as parameter...")
    capacity = 6
    nservers = 1
    queue = GenericQueue(capacity, nservers)
    env = EnvQueueSingleBufferWithJobClasses(queue, [0.3, 0.8, 0.7, 0.9], rewardOnJobClassAcceptance, [1, 0.8, 0.3, 0.2])
    K = env.getCapacity()
    J = env.getNumJobClasses()
    R = env.getRewardsForJobClassAcceptance()
    nS = len(env.getActions())

    # -- Equivalent job arrival rates (by server)
    print("Job arrival rates: {}".format(env.job_class_rates))
    print("Equivalent arrival rates by server: {}".format(env.job_rates_by_server))
    assert env.job_rates_by_server == [np.sum(env.job_class_rates)]

    # -- Arriving job on buffer not full
    buffer_size_orig = 0
    job_class = 1
    env.setJobClass(job_class)
    state = env.getState()
    assert state == (0, job_class)
    # Action = reject
    next_state, reward, done, info = env.step(0)
    assert next_state == (0, None)
    assert reward == 0
    # Action = accept
    env.setJobClass(job_class)
    next_state, reward, done, info = env.step(1)
    assert next_state == (buffer_size_orig + 1, None)
    assert reward == env.getRewardForJobClassAcceptance(job_class)

    # -- Arriving job on full buffer
    env.setBufferSize(env.getCapacity())
    job_class = 1
    env.setJobClass(job_class)
    state = env.getState()
    assert state == (env.getCapacity(), job_class)
    # Action = reject
    next_state, reward, done, info = env.step(0)
    assert next_state == (env.getCapacity(), None)
    assert reward == 0
    # Action = accept (not taken because buffer is full)
    env.setJobClass(job_class)
    next_state, reward, done, info = env.step(1)
    assert next_state == (env.getCapacity(), None)
    assert reward == 0
