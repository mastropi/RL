# -*- coding: utf-8 -*-
"""
Created on 19 Oct 2021

@author: Daniel Mastropetro
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
