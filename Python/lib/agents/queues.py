# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:43:28 2021

@author: Daniel Mastropietro
@description: Definition of agents acting on queue environments
"""

from enum import Enum, unique

if __name__ == "__main__":
    import runpy
    runpy.run_path("../../../setup.py")

from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.policies import PolicyTypes
from Python.lib.environments.queues import ActionTypes, EnvQueueSingleBufferWithJobClasses


class AgeQueue(GenericAgent):
    """
    Agent that acts on a queue environment

    Arguments:
    env: environment
        Queue environment defining the characteristics of the environment on which the agent acts.
        Note that this object is NOT stored as part of the object's attributes, and the reason is that we may want
        the agent to act on any given number of *different* environments having the same characteristics
        (e.g. Fleming-Viot learning, where there are N copies of the same environment on which the same type of agent
        interacts --by e.g. accepting or rejecting incoming jobs).
        Here the queue environment is only used to retrieve the number of servers in the system and the arrival rates
        of the different arriving job classes, in order to define the job assignment policy
        (of an incoming job of a given class to a server in the queue system).

    policies: dict
        Dictionary of policies followed by the agent, with the following elements:
        - PolicyTypes.ACCEPT: Policy object defining the acceptance policy of an incoming job. The acceptance policy
        may depend on the class of the incoming job. If so, the policy object defined by this dictionary key should be
        a list of policies with as many elements as job classes accepted by the queue environment on which the policy
        is applied.
        - PolicyTypes.ASSIGN: Policy object defining the assignment policy of an incoming job class to a server.
            The assignment policy can be None, in which case the assignment of job classes to servers follows a uniform probability
            over all the servers in the queue.

    learners: dict
        Dictionary of learners, with the following elements:
        'V': GenericLearner used by the agent to learn the state value function.
        'Q': GenericLearner used by the agent to learn the action-state value function.
    """
    def __init__(self, env, policies: dict, learners: dict, debug=False):
        super().__init__(policies, learners)
        self.debug = debug

        # Check whether the acceptance policy depends on the arriving job class or is the same for all job classes
        # We assume the policy depends on the job class when the `policies` parameter is a list of policies with
        # as many elements as job classes on which the policy is applied.
        if isinstance(self.getPolicy()[PolicyTypes.ACCEPT], list):
            self.acceptance_policy_depends_on_jobclass = True
        else:
            self.acceptance_policy_depends_on_jobclass = False

    def act(self, env, policy_type):
        """
        The agent performs an action that is chosen by the policy of the given type.
        The action performed is stored in the object and can be retrieved with the getLastAction() method.

        Arguments:
        env: Queue environment (normally defined in environments/queues.py)
            The queue environment on which the agent acts.
            NOTE: This object is updated after the action takes place.

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

        job_class = env.getJobClass()

        if policy_type == PolicyTypes.ACCEPT:
            # Choose the action to take and store it in the agent object
            action_type = ActionTypes.ACCEPT_REJECT
            if self.acceptance_policy_depends_on_jobclass:
                # This is most likely the case when the environment is a loss network with multi-class jobs
                assert job_class < len(env.getQueueState())
                action = self.getAcceptancePolicies()[job_class].choose_action((env.getQueueState()[job_class], job_class))
                ## IMPORTANT: The state that is passed to the policy's choose_action() method MUST be the state of the queue environment on which the policy acts
                ## So for instance, for queue environments defined in environments/queues.py (such as EnvQueueSingleBufferWithJobClasses or EnvQueueLossNetworkWithJobClasses)
                ## the state is a tuple containing the queue's state and the arriving job class.
                ## The message here is that we should NOT simply pass the queue's state as parameter to choose_action() but also
                ## the information of the arriving job class, as the queue's state is extacted from such tuple (queue_state, job_class)
                ## by the getPolicyForAction() method of the parameterized policy that is called by the choose_action method called above.

                ## NOTE also that, as queue_state we pass the occupancy level of the arriving job class because this is the piece of information of the state
                ## that defines the acceptance or rejection of the job... i.e. the occupancy level of the other job classes do not play any role in this acceptance policy.
            else:
                # This is most likely the case when the environment is a single- or multi-server system with single queue (a.k.a. buffer)
                action = self.getAcceptancePolicies()[0].choose_action(env.getState())
            self.setLastAction(action)
        elif policy_type == PolicyTypes.ASSIGN:
            # Choose the action to take and store it in the agent object
            # In an ASSIGN policy type the action is the server number to which the arriving job is assigned
            if env.getNumServers() > 1:
                assert self.getAssignmentPolicy() is not None, "The assignment policy must be defined in multi-server systems"
                servers = range(env.getNumServers())
                # TODO: (2021/10/21) Homogenize the signature of choose_action() among policies so that they all receive one parameter: the current state (but I am not sure whether this is possible)
                action = self.getAssignmentPolicy().choose_action(job_class, servers)
            else:
                # The action is choose the only server in the system, server #0
                action = 0
            action_type = ActionTypes.ASSIGN
            self.setLastAction(action)

        # Take the action, observe the next state, and get the reward received when transitioning to that state
        observation, reward, info = env.step(action, action_type)

        # Update the action taken by the agent after the environment performed the step triggered by the action chosen by the agent above
        # Note that the action can change after the environment received it from the agent and tried to make a step consistent with that action because,
        # that action was actually not possible to be performed given the environment conditions.
        # This is the case for instance when a policy (that depends on the job class --i.e. self.acceptance_policy_depends_on_jobclass = True)
        # accepts a job based solely on the server occupancy of THAT ARRIVING JOB CLASS, but doesn't know anything about the system's occupancy
        # (when the occupancy of all job classes in the system are taken into account), and the system's occupancy happens to be at full capacity!
        # In such case, the job is actually rejected (by the environment, not by the policy). So here we update the action actually taken by the agent
        # (rejection instead of its initial approval)
        # Impacting the agent's action in this way is important, because o.w. an assertion written in the manage_job_arrival() function
        # in simulators/queues.py would fail (because the assertion asserts that the reward is 0.0 when the action is ACCEPT, which would NOT be the case
        # should the agent's action not be updated as done here.
        # Note however that the last action taken by the agent (stored in the agent object) is NOT updated... so we can still know
        # what action the agent took by retrieving the agent's last action with agent.getLastAction().
        action = env.getLastAction()

        return action, observation, reward, info

    def learn(self, t, state, reward, done, info):
        # TODO: (2021/10/20) Should the agent learn or should we leave this implementation to the simulator that learns? (see e.g. simulators.py/SimulatorQueue)
        pass

    #------ GETTERS ------#
    def getAcceptancePolicies(self):
        policies_accept = self.getPolicy()[PolicyTypes.ACCEPT]
        if self.acceptance_policy_depends_on_jobclass:
            # The agent defines several acceptance policies, one per arriving job class
            # (which of course could be just one job class, in which case still the policy is stored as a list --of one policy)
            # => Return all the acceptance policies defined in the agent
            return policies_accept
        else:
            # There is only one policy defined in the agent
            # => Convert the acceptance policy to a list containing the acceptance policy so that callers to this method
            # have to deal only with one case, namely the case of *multiple* policies stored in a list.
            return [policies_accept]

    def getAcceptancePolicyDependenceOnJobClass(self):
        return self.acceptance_policy_depends_on_jobclass

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
