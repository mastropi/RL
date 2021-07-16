# -*- coding: utf-8 -*-
"""
Created on 16 Jul 2021

@author: Daniel Mastropetro
@description: Definition of environments on queues (without using the gym module)
"""

from warnings import warn
import numpy as np

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../../setup.py')

    from Python.lib.queues import GenericQueue, QueueMM
else:
    from ..queues import GenericQueue, QueueMM

class EnvQueueSingleBufferWithJobClasses:
    """
    Queue environment with a single buffer receiving jobs of different classes
    being served by a one or multiple servers.

    Arguments:
    queue: QueueMM
        The queue object that governs the dynamics of the environment through the definition of a server system.

    job_rates: list
        A list containing the arriving job rates for each valid job class.

    rewards: list
        List of rewards for each job class whose rate is given in job_rates.

    policy_assign: (opt) list of lists
        Assignment policy defined as a list of assignment probabilities for each job class
        (associated to job rates) to a server in the queue.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        default: None (in which case the assignment probability is uniform over the servers)
    """

    def __init__(self, queue :QueueMM, job_rates :list, rewards :list, policy_assign :list=None):
        if len(rewards) != len(job_rates):
            raise ValueError("The number of rewards ({}) must be the same as the number of job rates ({})".format(len(rewards), len(job_rates)))

        self.queue = queue
        self.job_rates = job_rates
        self.rewards = rewards

        # Job-class to server assignment policy
        if policy_assign is None:
            # When no assignment probability is given, it is defined to be uniform over the servers for each job class
            policy_assign = [ [1.0 / self.queue.getNServers()] * self.queue.getNServers() ] * len(self.job_rates) 
        self.policy_assign = policy_assign

        # Job-rates by server (assuming they are pre-assigned)
        self.job_rates_by_server = self.compute_arrival_rates_by_server()

        self.reset()

    def reset(self):
        self.buffer_size = 0
        self.job_class = None

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
        J = len(self.job_rates)         # Number of job classes
        job_arrival_rates = [0]*R
        for r in range(R):
            for c in range(J):
                job_arrival_rates[r] += self.policy_assign[c][r] * self.job_rates[c]

        return job_arrival_rates

    #------ GETTERS ------#
    def getCapacity(self):
        return self.queue.getCapacity()

    def getJobRates(self):
        return self.job_rates

    def getJobRatesByServer(self):
        return self.job_rates_by_server

    def getServiceRates(self):
        return self.queue.getDeathRates()

    def getIntensities(self):
        return [b/d for b, d in zip(self.getJobRatesByServer(), self.getServiceRates())]

    def getAssignPolicy(self):
        return self.policy_assign 

    def getNumJobClasses(self):
        return len(self.job_rates)

    def getNumServers(self):
        return self.queue.getNServers()

    def getRewards(self):
        return self.rewards

    def getState(self):
        return (self.buffer_size, self.job_class)

    #------ SETTERS ------#
    def setBufferSize(self, buffer_size :int):
        "Sets the buffer size of the queue"
        if int(buffer_size) != buffer_size or buffer_size < 0 or buffer_size > self.getCapacity():
            raise ValueError("The buffer size value is invalid ({}). It must be an integer between {} and {}".format(buffer_size, 0, self.getCapacity()))
        self.buffer_size = buffer_size

    def setJobClass(self, job_class):
        "Sets the job class of a new arriving job"
        if int(job_class) != job_class or job_class < 0 or job_class >= self.getNumJobClasses():
            raise ValueError("The job class is invalid ({}). It must be an integer between {} and {}".format(job_class, 0, self.getNumJobClasses()-1))
        self.job_class = job_class
