# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:43:28 2021

@author: Daniel Mastropietro
@description: Definition of agents acting on queue environments
"""

from enum import Enum, unique

from environments.queues import ActionTypes
from . import GenericAgent
from .policies.job_assignment import PolJobAssignmentProbabilistic


@unique
class PolicyTypes(Enum):
    ACCEPT = 'accept'
    ASSIGN = 'assign'

@unique
class LearnerTypes(Enum):
    V = 'state_value'
    Q = 'action_state_value'
    P = 'policy'


class AgeQueue(GenericAgent):
    """
    Agent that acts on a queue environment

    Arguments:
    env: Queue environment (normally defined in environments/queues.py)
        The environment where the agent acts.

    policies: dict
        Dictionary of policies followed by the agent, with the following elements:
        - PolicyTypes.ACCEPT: Policy object defining the acceptance policy of an incoming job.
        - PolicyTypes.ASSIGN: Policy object defining the assignment policy of an incoming job class to a server.
            The assignment policy can be None, in which case the assignment of job classes to servers follows a uniform probability
            over all the servers in the queue.

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
            # No assignment policy is given
            # # => Define it as a policy with uniform probability over servers for each job class
            self.setAssignmentPolicy( PolJobAssignmentProbabilistic(
                [[1.0 / self.env.getNumServers()] * self.env.getNumServers()] * len(self.env.getJobClassRates()) ) )

    def act(self, policy_type):
        """
        The agent performs an action that is chosen by the policy of the given type.
        The action performed is stored in the object and can be retrieved with the getLastAction() method.

        Arguments:
        policy_type: PolicyTypes
            Type of policy on which the agent acts on the queue, one of:
            - PolicyType.ACCEPT which is used to decide whether an incoming job is accepted to the queue
            - PolicyType.ASSIGN which is used to decide to which server a the class of an accepted incoming job is assigned

        Return: tuple
        Tuple containing the following elements:
        - action: the action taken by the agent on the given policy
        - observation: the next state on which the queue transitions to after the action taken
        - reward: the reward received by the agent after taking the action and transitioning to the next state
        - info: dictionary with relevant additional information
        """
        if not self.isValidPolicyType(policy_type):
            raise ValueError("Invalid policy type to act on. Possible types are: {} ({})" \
                             .format(PolicyTypes, policy_type))

        if policy_type == PolicyTypes.ACCEPT:
            # Choose the action to take and store it in the agent object
            action_type = ActionTypes.ACCEPT_REJECT
            action = self.getPolicy()[PolicyTypes.ACCEPT].choose_action()
            self.setLastAction(action)
        elif policy_type == PolicyTypes.ASSIGN:
            # Choose the action to take and store it in the agent object
            action_type = ActionTypes.ASSIGN
            job_class = self.env.getJobClass()
            servers = range(self.env.getNumServers())
            action = self.getPolicy()[PolicyTypes.ASSIGN].choose_action(job_class, servers)
            self.setLastAction(action)

        # Take the action, observe the next state, and get the reward received when transitioning to that state
        observation, reward, info = self.env.step(action, action_type)

        return action, observation, reward, info

    def learn(self, t, state, reward, done, info):
        # TODO: (2021/10/20) Should the agent learn or should we leave this implementation to the simulator that learns? (see e.g. simulators.py/SimulatorQueue)
        pass

    #------ GETTERS ------#
    def getAcceptancePolicy(self):
        return self.getPolicy()[PolicyTypes.ACCEPT]

    def getAssignmentPolicy(self):
        return self.getPolicy()[PolicyTypes.ASSIGN]

    def getLearnerV(self):
        return self.getLearner()[LearnerTypes.V]

    def getLearnerQ(self):
        return self.getLearner()[LearnerTypes.Q]

    def getLearnerP(self):
        return self.getLearner()[LearnerTypes.P]

    #------ SETTERS ------#
    def setAssignmentPolicy(self, policy_assign):
        self.getPolicy()[PolicyTypes.ASSIGN] = policy_assign

    #------ CHECKERS ------#
    def isValidPolicyType(self, policy_type):
        return policy_type in PolicyTypes