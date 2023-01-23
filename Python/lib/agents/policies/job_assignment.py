# -*- coding: utf-8 -*-
"""
Created on 19 Oct 2021

@author: Daniel Mastropietro
@description: Definition of an assignment policy of jobs to servers to be used in queues
"""

import numpy as np


class PolJobAssignmentProbabilistic:
    """
    Job assignment policy based on a probabilistic mapping from job classes to servers in the queue.

    Arguments:
    policy_assign: Policy
        Policy object specifying the assignment policy to use to assign a new job class to a server in the queue
        based on a probabilistic approach.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        PolJobAssignmentProbabilistic( [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
    """

    def __init__(self, policy_assign :list):
        self.prob_map_from_jobclass_to_server = policy_assign

    def getProbabilisticMap(self):
        return self.prob_map_from_jobclass_to_server

    def getProbabilisticMapForJobClass(self, job_class):
        return self.prob_map_from_jobclass_to_server[job_class]

    def choose_action(self, job_class, servers):
        server_assigned_to_job = np.random.choice(servers, p=self.getProbabilisticMapForJobClass(job_class))
        return server_assigned_to_job


def define_uniform_job_assignment_policy(num_job_classes: int, num_servers: int):
    """
    Defines a policy with uniform probability of assignment to servers for each job class

    Arguments:
    num_job_classes: int
        Number of possible job classes to assign to a server.

    num_servers: int
        Number of available servers to assign a job to.

    Return: PolJobAssignmentProbabilistic object
    A job-assignment-to-server policy that assigns each job class with uniform probability to each possible server.
    """
    return PolJobAssignmentProbabilistic([[1.0 / num_servers] * num_servers] * num_job_classes)
