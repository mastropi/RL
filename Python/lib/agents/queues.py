# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:43:28 2021

@author: Daniel Mastropietro
@description: Definition of agents acting on queue environments
"""

from enum import Enum, unique

from . import GenericAgent

@unique
class PolicyTypes(Enum):
    ACCEPT = 'accept'
    ASSIGN = 'assign'


class AgeQueue(GenericAgent):
    """
    Agent that acts on a queue environment

    Arguments:
    env: Queue environment (normally defined in environments/queues.py)
        The environment where the agent acts.

    policies: dict
        Dictionary of policies followed by the agent, with the following elements:
        'accept': Policy object defining the acceptance policy of an incoming job.
        'assign': List of lists defining the assignment policy of an incoming type of job to a server in the queue.
            It is defined as a list of assignment probabilities for each type of incoming job.
            The list at the top level is indexed by the type of incoming job.
            The list at the second level is indexed by the server in the queue.

            Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
            to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

            default: None (in which case the assignment probability is uniform over the servers)

    learners: dict
        Dictionary of learners, with the following elements:
        'V': Learner used by the agent to learn the state value function.
        'Q': Learner used by the agent to learn the action-state value function.
    """
    def __init__(self, env, policies: dict, learners: dict,
                 debug=False):
        super().__init__(policies, learners)
        self.debug = debug

        # Environment on which the agent acts
        self.env = env

        # Define the default job-type-to-server assignment policy in case it is None
        if self.getAssignmentPolicy() is None:
            # No assignment probability is given
            # # => Define it as a policy with uniform probability over servers for each job class
            self.setAssignmentPolicy( [[1.0 / self.env.getQueue().getNServers()] * self.env.getQueue().getNServers()] * len(self.env.getJobClassRates()) )

    def act(self, policy_type='accept'):
        "The agent performs an action that is chosen by the policy of the given type"
        if not self.isValidPolicyType(policy_type):
            raise ValueError("Invalid policy type to act on. Possible types are: {} ({})" \
                             .format(PolicyTypes.values, policy_type))

        action = self.getPolicy()[policy_type].choose_action()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def learn(self, state, reward, done, info):
        # TODO
        return 0.0

    #------ GETTERS ------#
    def getAcceptancePolicy(self):
        return self.getPolicy()[PolicyTypes.ACCEPT]

    def getAssignmentPolicy(self):
        return self.getPolicy()[PolicyTypes.ASSIGN]

    #------ SETTERS ------#
    def setAssignmentPolicy(self, policy_assign):
        self.getPolicy()[PolicyTypes.ASSIGN] = policy_assign

    #------ CHECKERS ------#
    def isValidPolicyType(self, policy_type):
        return policy_type in PolicyTypes.values
