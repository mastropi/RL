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
from Python.lib.environments.queues import Actions, ActionTypes


class AgeQueue(GenericAgent):
    """
    Agent that acts on a queue environment

    Arguments:
    env: Queue environment
        The queue environment on which the agent acts is NOT stored in the class because we may want
        the agent to act on *different* environments having the same or similar characteristics
        (e.g. in Fleming-Viot learning, where there are N copies of the same environment on which the same type of agent
        interacts --by e.g. accepting or rejecting incoming jobs).

        The attribute is left as part of the constructor because, conceptually, agents should receive the environment on
        which they act, so that we homogenize the signature of all agents we define, by passing the environment,
        the policies, and the learners.

        The queue environment on which the agent acts is passed as argument to the act() method defined in the class.

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
        'Q': GenericLearner used by the agent to learn the action value function.
    """
    def __init__(self, env, policies: dict, learners: dict, debug=False):
        super().__init__(policies, learners)
        self.debug = debug

        # Check whether the acceptance policy depends on the arriving job class or is the same for all job classes
        # We assume the policy depends on the job class when the `policies` parameter is a list of policies with
        # as many elements as job classes on which the policy is applied.
        # NOTE that we cannot call the self.getAcceptancePolicies() method defined below because that method USES
        # the value of the attribute we define here self.acceptance_policy_depends_on_jobclass whose value we set now!
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
            It should have the following methods defined:
            - getActions(): gets the possible actions accepted by the environment.
            - getBufferSize(): gets the current queue length, i.e. the sum of the occupancy levels of each job class
                received by the queue system defined in the environment.
            - getCapacity(): gets the capacity of the queue system defined in the environment.
            - getJobClass(): gets the job class associated to the current state of the environment. If not None
                this is the class of the arriving job at the time an action is required by the agent.
            - getLastAction(): get the last action taken by the agent on the environment.
            - getNumServers(): gets the number of servers in the queue system defined in the environment.
            - getQueueState(): gets the queue state of the queue system defined in the environment.
            - getState(): gets the current state of the environment.
            - step(): the method that takes the action chosen by the agent on the environment.
            NOTE: This object is updated after the action takes place with the result of taking the action of the given
            policy_type chosen by the agent.

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
                # Note that the action of the agent is REJECT when the system is at FULL CAPACITY (regardless of the class of the job that arrives)
                # If the system is NOT at full capacity, the accept/reject action is chosen based on the class of the arriving job
                # and the occupancy level of that class in the system, which are inputs for the acceptance policy.
                assert job_class < len(env.getQueueState()), "The job class ({}) must be an index whose value is less than the queue state length ({})".format(job_class, len(env.getQueueState()))
                buffer_size_is_smaller_than_queue_capacity = env.getBufferSize() < env.getCapacity()
                # The action taken is the minimum between two possible actions:
                # - accepting/rejecting the job based on the system being at full capacity or not.
                # - accepting/rejecting the job based on the policy for the class of the arriving job.
                # The fact that we use the minimum assumes that the possible actions defined in the Actions enum
                # associates REJECT to a smaller index than ACCEPT, and in addition the way to retrieve the chosen action
                # requires that REJECT is associated to index 0 and ACCEPT is associated to index 1 (because the chosen
                # action is derived from the above boolean expression which is converted to integer on the next line).
                assert Actions.ACCEPT.value > Actions.REJECT.value, "The REJECT action ({}) must have a lower index than the ACCEPT action ({}) in the Actions enum".format(Actions.REJECT, Actions.ACCEPT)
                assert Actions.REJECT.value == 0 and Actions.ACCEPT.value == 1, "The REJECT action ({}) must have index value 0 and the ACCEPT action ({}) must have index value 1 in the Actions enum".format(Actions.REJECT, Actions.ACCEPT)
                action = min( env.getActions()(int(buffer_size_is_smaller_than_queue_capacity)),
                              self.getAcceptancePolicies()[job_class].choose_action((env.getQueueState()[job_class], job_class)) )
                    ## IMPORTANT: The state that is passed to the policy's choose_action() method MUST be the state of the queue environment on which the policy acts
                    ## So for instance, for queue environments defined in environments/queues.py (such as EnvQueueSingleBufferWithJobClasses or EnvQueueLossNetworkWithJobClasses)
                    ## the state is a tuple containing the queue's state and the arriving job class.
                    ## The message here is that we should NOT simply pass the queue's state as parameter to choose_action() but also
                    ## the information of the arriving job class, as the queue's state is extacted from such tuple (queue_state, job_class)
                    ## by the getPolicyForAction() method of the parameterized policy that is called by the choose_action method called above.

                    ## NOTE also that, as queue_state we pass the occupancy level of the arriving job class because this is the piece of information of the state
                    ## that defines the acceptance or rejection of the job... i.e. the occupancy level of the other job classes do not play any role in this acceptance policy.
                assert isinstance(action, Actions)
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
        "Returns a *list* with the acceptance policies defined in the agent, either when only one policy is defined or when multiple policies are defined"
        policies_accept = self.getPolicy()[PolicyTypes.ACCEPT]
        if self.acceptance_policy_depends_on_jobclass:
            # The agent defines several acceptance policies, e.g. one per arriving job class
            # (which of course could be just one job class, in which case still the policy is stored as a list --of one policy)
            # => Return all the acceptance policies defined in the agent
            return policies_accept
        else:
            # There is only one policy defined in the agent
            # => Convert the acceptance policy to a list containing the acceptance policy so that callers to this method
            # have to deal only with one case, namely the case of *multiple* policies stored in a list.
            return [policies_accept]

    def getAcceptancePolicyThresholds(self):
        "Returns the thresholds (theta parameters) of the acceptance policies defined in the agent. Each acceptance policy is expected to have a SCALAR theta parameter, that's why we use the word 'threshold' in the method's name."
        return [pol.getThetaParameter() for pol in self.getAcceptancePolicies()]

    def getAcceptancePolicyIntegerThresholds(self):
        thresholds = self.getAcceptancePolicyThresholds()
        policies_accept = self.getAcceptancePolicies()
        assert isinstance(thresholds, list)
        assert isinstance(policies_accept, list)
        return [policies_accept[p].getDeterministicBlockingValueFromTheta(theta) for p, theta in enumerate(thresholds)]

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
    def setAcceptancePolicyThresholds(self, thresholds):
        policies_accept = self.getAcceptancePolicies()
        # Set the threshold of each acceptance policy
        if thresholds is None or len(thresholds) != len(policies_accept):
            raise ValueError("The `thresholds` parameter must not be None and its length must be equal to the number of policies defined in the agent")
        for i, policy in enumerate(policies_accept):
            policy.setThetaParameter(thresholds[i])

    def setAssignmentPolicy(self, policy_assign):
        # Note that we cannot invoke the getAssignmentPolicy() method defined above
        # because it is not possible to assign a value to a function call (this is the error we get when we try to use that method on the LHS of the equality)
        self.getPolicy()[PolicyTypes.ASSIGN] = policy_assign

    #------ CHECKERS ------#
    def isValidPolicyType(self, policy_type):
        return policy_type in PolicyTypes
