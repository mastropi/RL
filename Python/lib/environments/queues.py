# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropietro
@description: Definition of queue environments.
                An environment should define the state space, the action space, and the rewards received for each
                action taken on a given state.

                In the particular case of queue environments, it should ALSO define the queue system that is responsible
                of generating the time and type of the next event.
                This allows for the simulation of an SMDP, i.e. a Semi-Markov Decision Process, where the times at which
                the MDP iterates are random (in this case, defined by the queue system behind it).

                Note that it should NOT define any agents acting on the environment, as this is done separately.
"""

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../../setup.py')

import copy
from enum import Enum, IntEnum, unique
from typing import Union

import numpy as np

import gym
from gym import spaces

from Python.lib.queues import QueueMM

# NOTE: We use IntEnum instead of Enum because of the problem of comparison described here!!
# https://stackoverflow.com/questions/28125055/enum-in-python-doesnt-work-as-expected
# The problem of using Enum has to do with the fact that the comparison between two Enum values
# (as in action = Actions.REJECT) is done via id, i.e. by their memory address --because Enum does NOT have
# the equal operator implemented, so it checks equality via their memory addresses.
# The memory address of each Enum value WILL be different from the memory address of that same Enum value
# if the same Enum has been imported differently... e.g. by using absolute paths vs. using relative paths...
# (as apparently Python doesn't recognize that the same Enum has been imported when importing it again using
# a relative path if the previous import was done with an absolute path --my guess here).
# This is explained in the answer by Austin Basye in the above link.
# BOTTOM LINE: Do not mix relative with absolute imports!! --> I SHOULD FIX THIS IN MY CODE! (2021/12/10)
@unique
class ActionTypes(IntEnum):
    ACCEPT_REJECT = 1
    ASSIGN = 2

@unique
class Actions(IntEnum):
    REJECT = 0
    ACCEPT = 1
    OTHER = -99      # This is used for actions of ActionTypes = ASSIGN (where the action is actually the server number to which the accepted job is assigned)

@unique
class BufferType(Enum):
    SINGLE = 1
    NOBUFFER = 2

# Default reference buffer size used in the definition of the exponential cost when blocking an incoming job
# This value is the minimum expected cost that is searched for by the learning process when only hard (i.e. deterministic) blocking is used)
COST_EXP_BUFFER_SIZE_REF = 40


#---------------------------- Reward functions that can be used in different environments -----------------------------#
# Each reward function should depend on 4 inputs:
# - env: the environment  where the reward is generated
# - S(t) the state at which an action is applied
# - A(t) the action applied at state S(t)
# - S(t+1) the next state where the system is at after applying action A(t) on state S(t)

def rewardOnJobClassAcceptance(env, state, action, next_state, dict_params=None):
    # The state is assumed to be a tuple (k, i) where k is the buffer size
    # (either the size of the queue's SINGLE buffer or the occupancy number of the job class i in the system)
    # and i is the class of the arriving job
    if action == Actions.ACCEPT:
        job_class = state[1]
        return env.getRewardForJobClassAcceptance(job_class)
    else:
        return 0.0

def rewardOnJobRejection_Constant(env, state, action, next_state, dict_params=None):
    "Negative reward received when an incoming job is rejected"
    if dict_params is None:
        dict_params = dict({'reward_at_rejection': -1.0})

    if action == Actions.REJECT:
        # Blocking occurred
        # => There is a negative reward or cost
        assert env.getBufferSizeFromState(state) == env.getBufferSizeFromState(next_state), \
            "At REJECT, the queue's buffer size after rejection ({}) is the same as before rejection ({})" \
            .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))
        reward = dict_params['reward_at_rejection']
        assert reward < 0.0, "The reward of job rejection is negative ({})".format(reward)
        return reward
    else:
        # No blocking
        # => No reward
        return 0.0

def rewardOnJobRejection_ExponentialCost(env, state, action, next_state, dict_params=None):
    "Negative reward received when an incoming job is rejected as a function of the queue's buffer size associated to the given state"
    if dict_params is None:
        dict_params = dict({'buffer_size_ref': COST_EXP_BUFFER_SIZE_REF})
    if 'buffer_size_ref' not in dict_params.keys():
        raise ValueError("Parameter 'buffer_size_ref' must be given in the parameters dictionary containing"
                         " the reference buffer size, above which there is an exponentially increasing cost with"
                         " the buffer size when a job is rejected.")

    if action == Actions.REJECT:
        # Blocking occurred
        # => There is a negative reward or cost
        assert env.getBufferSizeFromState(state) == env.getBufferSizeFromState(next_state), \
            "At REJECT, the queue's buffer size after rejection ({}) is the same as before rejection ({})" \
            .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))
        reward = -costBlockingExponential(env.getBufferSizeFromState(state), dict_params['buffer_size_ref'])
        assert reward < 0.0, "The reward of job rejection is negative ({})".format(reward)
        return reward
    else:
        # No blocking
        # => No reward
        return 0.0

def costBlockingExponential(buffer_size: int, buffer_size_ref :int):
    """
    Cost of blocking a job from entering a queue, modelled as exponentially increasing on the queue's buffer size
    with respect to a minimum buffer size, below which the cost is constant.
    """
    # C(s, a=block)
    # The expected cost function is:
    #   E(C(s,a)) = E( C(S(t), a=block) * I{A(t)=block} ) = ( B I{s<=sref} + B*b**(s-sref) I{s>sref} ) * Pr(A(t)=block)
    #
    # *****************************************************************************************************************
    # NOTE: (2022/01/24) WE COULD HAVE CONSIDERED A COST FUNCTION THAT IS AN EXPONENTIALLY INCREASING FUNCTION WITH s!!
    # (as proposed originally by Matt), i.e.:
    #   C(s,a) = B + B*b**(s-sref)
    # i.e. without indicator functions and which is equal to B*(1+b**(-sref)) at s=0, and is equal to 2B at s=sref
    # and sref controls the location of the minimum!
    # In fact, in that case we would get the following expected cost:
    #   E(C(s,a)) = B * (1 + b**(s-sref)) * Pr(A(t)=block)
    # which has a minimum near sref, more precisely, if sref is large (making 1 - rho**(sref+1) ~= 1), the minimum is @:
    #   sref - shift
    # where the shift is a positive shift equal to:
    #   shift = log( -log(rho) / (log(b) + log(rho)) ) / log(b)
    # which is a controlled shift (i.e. it doesn't go to infinity, as long as b is chosen larger away from 1/rho.
    # (e.g. b = 3.0 for rho = 0.7 suffices to get the shift equal to 0.666667)
    # (I wrote this in my orange Carrefour tiny notebook)
    # *****************************************************************************************************************
    #
    # Going to the originally defined cost function, we have the following graph for the expected cost:
    #   B = 1; b = 3.0; sref = 3;
    #   rho = 0.7
    #   s = np.linspace(0, 12, 100)
    #   indicator = 1.0 * ( s > sref )
    #   p = rho**s * (1 - rho) / (1 - rho**(s+1))
    #   f = (B * (1 - indicator) + B*b**(s - sref) * indicator) * p
    #   plt.plot(s, f, 'b-');
    #   ax = plt.gca(); ax.set_ylim((0,2))
    B = 5.0      # Blocking associated just to the fact the queue is blocked
    b = 3.0     # Base of the exponential function
    #cost = B if buffer_size <= buffer_size_ref else B * b**(buffer_size - buffer_size_ref)
    cost = B * (1 + b**(buffer_size - buffer_size_ref))
    if False:
        print("Computing cost for buffer size = {} with sref = {} --> Cost = {:.3f}".format(buffer_size, buffer_size_ref, cost))
    if False and cost != B:
        print("The cost is in the exponential part (increasing with `buffer size - REF buffer size`): buffer={}, ref={}".format(buffer_size, buffer_size_ref))
    return cost
#---------------------------- Reward functions that can be used in different environments -----------------------------#


class GenericEnvQueueWithJobClasses(gym.Env):
    """
    Class defining methods that are generic to queue environments receiving multi-class jobs.

    Arguments:
    queue: QueueMM
        The queue object that governs the dynamics of the environment through the definition of a server system.

    buffer_type: BufferType
        The type of buffer in the queue. Any valid values of the BufferType enum.

    reward_func: function
        Function returning the reward received after taking action A(t) at state S(t) and making the environment
        go to state S(t+1).

    rewards_accept_by_job_class: list
        List of rewards for the acceptance of the different job class whose rate is given in job_class_rates.

    dict_params_reward_func: (opt) dict
        Dictionary containing non-standard parameters (i.e. besides the state, the action and the next state)
        on which the reward function `reward_func` depends on.
        Ex: dict_params_reward_func = {'buffer_size_ref': 20}
        default: None, in which case the default parameter values are used
    """

    def __init__(self, queue: QueueMM, buffer_type: BufferType,
                 reward_func, rewards_accept_by_job_class: list,
                 dict_params_reward_func: dict=None):
        self.queue = queue
        self.buffer_type = buffer_type

        #-- Rewards
        self.reward_func = reward_func
        self.dict_params_reward_func = dict_params_reward_func
        self.rewards_accept_by_job_class = rewards_accept_by_job_class

        # Create temporary variables that are shorter to reference and still easy to understand
        K = self.queue.getCapacity()

        # Action and observation spaces
        # TODO: (2021/10/22) Make sure we use these action and observation spaces in the implementation of actions and spaces in the environment
        # It seems to me that currently this information is never used.
        #self.action_space = spaces.Discrete(2)  # Either Reject (0) or Accept (1) an arriving job class
        self.action_space = Actions
        # DM-2023/01/03: The following is commented out because job_class_rates is not defined for all subclasses of this superclass
        # And in any case we are not using the observation_space defined here yet.
        #J = len(self.job_class_rates)
        #self.observation_space = spaces.Tuple((spaces.Discrete(K+1), spaces.Discrete(J)))

        # Last action taken by the agent interacting with the environment
        # (this piece of information is needed e.g. when learning the policy of the agent interacting
        # with the environment. See e.g. LeaPolicyGradient.learn())
        self.action = None

        # Seed used in the generation of random numbers during the interaction with the environment.
        # For instance to decide on blocking in a parameterized policy where theta defines a state with
        # RANDOM policy of acceptance (see e.g. GenericParameterizedPolicyTwoActions.choose_action())
        self.seed = None

        # Initialize the state of the system at 0 jobs of each class and no arriving job
        self.job_class = None  # Class of the arriving job (when a job arrives)
        start_state = (tuple([0] * self.queue.getNServers()), self.job_class)
        self.reset(start_state)

    # THIS METHOD COULD GO TO A SUPERCLASS
    def reset(self, state: tuple=None):
        """
        Resets the state of the environment and the last action it received

        Arguments:
        state: (opt) duple
            Duple containing the following information:
            - server sizes: list containing the queue size of each server making up the queue system
            - job_class: class of the arriving job
            default: None, in which case the queue is reset to the state with all 0 as server sizes and None job class
        """
        if state is None:
            state = (tuple([0] * self.queue.getNServers()), None)
        self.setState(state)
        self.action = None

    def reset(self, state: tuple=None):
        """
        Resets the state of the environment and the last action it received

        Arguments:
        state: (opt) duple
            Duple containing the following information:
            - server sizes: tuple with the size of each queue in each server making up the queue system
            - job_class: class of the arriving job
            default: None, in which case the queue is reset to the state with all 0s in the servers and None job class
        """
        if state is None:
            state = (tuple([0] * self.queue.getNServers()), None)
        self.setState(state)
        self.action = None

    def set_seed(self, seed):
        """
        Sets the seed of the environment if not None. If seed=None, no change is done to the random number generator.

        Note that this method is implemented because, although the gym.Env environment from which this class inherits
        has a seed() method, it doesn't work as it yields the following error when called as:
        env = gym.Env()
        env.seed()
        "UserWarning: WARN: Could not seed environment <Env instance>"

        with no further explanations!
        Python-3.6.4 (22-Oct-2021)
        """
        if seed != None:
            self.seed = seed
            np.random.seed(self.seed)

    def np_random(self):
        """
        Returns a number between 0 and 1 using the np.random.random() function.
        In order to set a seed for this random number generation, use the seed() method beforehand.
        """
        return np.random.random()

    #------ GETTERS ------#
    def get_seed(self):
        return self.getSeed()
    def getSeed(self):
        return self.seed

    def getQueue(self):
        return self.queue

    def getBufferType(self):
        return self.buffer_type

    def getCapacity(self):
        return self.queue.getCapacity()

    def getNumServers(self):
        return self.queue.getNServers()

    def getServiceRates(self):
        return list(self.queue.getDeathRates())

    def getQueueState(self):
        return tuple(self.queue.getServerSizes())

    def getQueueStateFromState(self, state: tuple):
        queue_state = state[0]
        return queue_state

    def getState(self):
        return self.state

    def getBufferSize(self):
        return self.queue.getBufferSize()

    def getBufferSizeFromState(self, state: tuple):
        queue_state = state[0]
        return np.sum(queue_state)

    def getJobClass(self):
        return self.job_class

    def getJobClassFromState(self, state: tuple):
        job_class = state[1]
        return job_class

    def getLastAction(self):
        return self.action

    def getActions(self):
        return self.action_space

    def getRewardsForJobClassAcceptance(self):
        return self.rewards_accept_by_job_class

    def getRewardForJobClassAcceptance(self, job_class: int):
        return self.rewards_accept_by_job_class[job_class]

    def getRewardFunction(self):
        return self.reward_func

    def getParamsRewardFunc(self):
        return self.dict_params_reward_func

    #------ SETTERS ------#
    def setCapacity(self, capacity):
        self.queue.setCapacity(capacity)

    def setParamsRewardFunc(self, dict_params):
        self.dict_params_reward_func = dict_params

    def setQueueState(self, queue_state: Union[int, tuple, np.ndarray]):
        self.queue.setServerSizes(queue_state)

    def setJobClass(self, job_class: int or None):
        "Sets the job class of a new arriving job and updates the environment's state. The job class can be None."
        if not (
                job_class is None or \
                int(job_class) == job_class and 0 <= job_class <= self.getNumJobClasses()
                ):
            raise ValueError(
                "The job class is invalid ({}). It must be an integer between {} and {}".format(job_class, 0, self.getNumJobClasses() - 1))
        self.job_class = job_class
        # Now we update the environment's state
        # NOTE that this call to self.setState() calls the appropriate method in the subclass of which `self` is an instance
        # (because the state is defined differently depending on the subclass).
        self.setState((self.getQueueState(), self.job_class))

    def setState(self, state: tuple):
        "This method is implemented by the subclass of which `self` is an instance"
        raise NotImplementedError

    def setLastAction(self, action):
        self.action = action


class EnvQueueSingleBufferWithJobClasses(GenericEnvQueueWithJobClasses):
    """
    Queue environment with a single buffer receiving jobs of different classes
    being served by one or multiple servers.

    The possible actions for an agent interacting with the queue are:
    - Accept or reject an incoming job of a given class --> action_type = ActionTypes.ACCEPT_REJECT
    - Choose the server to which the accepted incoming job is assigned --> action_type = ActionTypes.ASSIGN

    Arguments:
    queue: QueueMM
        The queue object that governs the dynamics of the environment through the definition of a server system.

    job_class_rates: list, tuple or numpy array
        A list or array containing the arriving job rates for each valid job class.
        It is always stored in the object as a list.

    reward_func: function
        Function returning the reward received after taking action A(t) at state S(t) and making the environment
        go to state S(t+1).

    rewards_accept_by_job_class: list
        List of rewards for the acceptance of the different job class whose rate is given in job_class_rates.

    dict_params_reward_func: (opt) dict
        Dictionary containing non-standard parameters (i.e. besides the state, the action and the next state)
        on which the reward function `reward_func` depends on.
        Ex: dict_params_reward_func = {'buffer_size_ref': 20}
        default: None, in which case the default parameter values are used
    """

    def __init__(self, queue: QueueMM, job_class_rates: list,
                 reward_func, rewards_accept_by_job_class: list,
                 dict_params_reward_func: dict=None):
        super().__init__(queue, BufferType.SINGLE, reward_func, rewards_accept_by_job_class, dict_params_reward_func)
        if not isinstance(job_class_rates, (list, np.ndarray)):
            raise ValueError("Input parameter `job_class_rates` must be a list or an array ({})".format(type(job_class_rates)))
        self.job_class_rates = list(job_class_rates)

        # State of the system
        self.state = (self.getBufferSize(), self.getJobClass())

    def step(self, action: int or Actions, action_type: ActionTypes):
        """
        The environment takes a step when receiving the given accept/reject action with the given assignment policy
        to assign the possibly accepted incoming job to a server in the queue system.

        Arguments:
        action: int or Actions
            Action taken by the agent interacting with the environment.
            If action_type == ActionTypes.ACCEPT_REJECT, the value should be of type Actions.
            If action_type == ActionTypes.ASSIGN, the value should be an integer, as it should indicate the server
            to which an incoming job is assigned.

        action_type: ActionTypes
            Type of the action taken (e.g. ActionTypes.ACCEPT_REJECT, ActionTypes.ASSIGN).

        Returns: tuple
        Tuple containing the following elements:
        - the next state where the environment went to
        - the reward received by taking the given action and transitioning from the current state to the next state
        - additional relevant information in the form of a dictionary
        """
        state = self.getState()
        if action_type == ActionTypes.ACCEPT_REJECT:
            assert self.getJobClass() is not None, "The job class of the arrived job is defined"

            # Convert the action to an ActionTypes class if given as integer
            if isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64):
                # Convert the action to an ActionTypes value
                action = Actions(action)

            if action not in [Actions.ACCEPT, Actions.REJECT]:
                raise ValueError("Invalid action: {}\nValid actions are: {}".format(action, Actions))

            if action == Actions.REJECT:
                # Reject the incoming job
                # => Just set the job class to None because no new job has yet arrived after taking the action
                # The state of the queue is left unchanged.
                self.setState((self.getQueueState(), None))
            elif action == Actions.ACCEPT:
                # => Accept the incoming job unless the queue's buffer is at full capacity
                # Note that when the job is accepted NOTHING is done here, because at this point
                # we don't know how to assign the accepted job to a server in the queue, as for that we need the
                # assignment policy, and this is a DIFFERENT policy, i.e. with actions of a different type
                # (the ActionsType.ASSIGN type).
                # So, accepting the job means simply "not rejecting it", and actually the ACTUAL acceptance
                # (i.e. the assignment of the job to a server) will be done separately when calling this step() function
                # with an ActionsType.ASSIGN type of action.
                if self.getBufferSize() == self.queue.getCapacity():
                    # Set the job class to None because no new job has yet arrived at the moment of taking the action
                    # The state of the queue is left unchanged.
                    self.setState((self.getQueueState(), None))
                    # Set the action to REJECT because the job was actually rejected, as the buffer was full!
                    action = Actions.REJECT
        elif action_type == ActionTypes.ASSIGN:
            # Current state of the queue system
            queue_state = self.getQueueState()
            # Define the next state of the queue system based on the action
            # (which represents the server to which an incoming job is assigned)
            queue_next_state = list(copy.deepcopy(queue_state))
            queue_next_state[action] += 1

            # Change the state of the system
            self.setState((tuple(queue_next_state), None))

            # Set the action to 'OTHER' because this is not an ACCEPT_REJECT type of action
            # This is important because otherwise the `reward_func` called below may interpret
            # the action as a REJECT or an ACCEPT if the action value is 0 or 1!
            action = Actions.OTHER

        # Define the next state and the reward of going to that next state
        # (This is a common task for ALL action types)
        next_state = self.getState()
        # Reward observed by going from S(t) to S(t+1) via action A(t)
        if self.reward_func is not None:
            reward = self.reward_func(self, state, action, next_state, dict_params=self.dict_params_reward_func)
        else:
            reward = 0.0

        # Update the last action stored in the object
        self.setLastAction(action)

        return next_state, reward, {}

    #------ GETTERS ------#
    def getJobClassRates(self):
        return self.job_class_rates

    def getNumJobClasses(self):
        return len(self.job_class_rates)

    #------ SETTERS ------#
    # TODO: (2023/01/26) I think we should set the state of the single-buffer queue system to the same state used in the loss network system so that this setState() method is NOT confusing...
    # In fact, now the `state` parameter has a different meaning than the state attribute of the class:
    # - the former refers to the tuple (queue_state, job_class)
    # - the latter refers to the tuple (buffer_size, job_class)
    # I propose that both states (the parameter and the attribute) be (queue_state, job_class) and that the buffer size is computed as needed from the queue state.
    # If we do this, the setState() could move to the super class (GenericEnvQueueWithJobClasses) as the two setState() currently defined will have the same signature.
    def setState(self, state):
        """
        Sets the state (k, i) of the queue environment from the server sizes and the arriving job class.
        - k is the system's buffer size, which is computed from the queue sizes of each server passed in parameter `state`
        - i is the arriving job class

        Arguments:
        state: duple
            Duple containing the following information:
            - server sizes: list with the size of each queue in each server making up the queue system
            - job_class: class of the arriving job
        """
        assert len(state) == 2
        # Extract the job class occupancy and the job class from the GIVEN state
        server_sizes, job_class = self.getQueueStateFromState(state), self.getJobClassFromState(state)
        self.setQueueState(server_sizes)
        self.job_class = job_class
        self.state = (self.getBufferSize(), self.job_class)


class EnvQueueLossNetworkWithJobClasses(GenericEnvQueueWithJobClasses):
    """
    Queue environment representing a loss network receiving jobs of different classes
    being served by multiple servers without buffer.

    The state of the system is the number of jobs of each class in the system, i.e. being served by one of the servers.

    The possible actions for an agent interacting with the queue are:
    - Accept or reject an incoming job of a given class --> action_type = ActionTypes.ACCEPT_REJECT

    Arguments:
    queue: QueueMM
        The queue object that governs the dynamics of the environment through the definition of a server system.
        In the M/M/c/K notation, c (usually the number of servers in the system) here is the number of different
        job classes, and K (usually the capacity of the system) here is the number of servers in the system that serve
        jobs.

    reward_func: function
        Function returning the reward received after taking action A(t) at state S(t) and making the environment
        go to state S(t+1).

    rewards_accept_by_job_class: list
        List of rewards for the acceptance of the different job class whose rate is given in job_class_rates.

    dict_params_reward_func: (opt) dict
        Dictionary containing non-standard parameters (i.e. besides the state, the action and the next state)
        on which the reward function `reward_func` depends on.
        Ex: dict_params_reward_func = {'buffer_size_ref': 20}
        default: None, in which case the default parameter values are used
    """

    def __init__(self, queue: QueueMM,
                 reward_func, rewards_accept_by_job_class: list,
                 dict_params_reward_func: dict=None):
        super().__init__(queue, BufferType.NOBUFFER, reward_func, rewards_accept_by_job_class, dict_params_reward_func)
        if self.queue.getNServers() != len(self.queue.getBirthRates()):
            raise ValueError("The number of servers in the queue ({}) must equal the number of job arrival rates ({}), "
                             "because the number of servers actually indicate the number of job classes." \
                             .format(self.queue.getNServers(), len(self.queue.getBirthRates)))
        # State of the system
        self.state = (self.getQueueState(), self.getJobClass())

    def step(self, action: Actions, action_type: ActionTypes=ActionTypes.ACCEPT_REJECT):
        """
        The environment takes a step when receiving the given accept/reject action

        Arguments:
        action: Actions
            Action taken by the agent interacting with the environment either accept or reject an incoming job.

        action_type: (opt) ActionTypes
            Type of the action taken. Only ActionTypes.ACCEPT_REJECT is accepted.
            default: ActionTypes.ACCEPT_REJECT

        Returns: tuple
        Tuple containing the following elements:
        - the next state where the environment went to
        - the reward received by taking the given action and transitioning from the current state to the next state
        - additional relevant information in the form of a dictionary
        """
        if action_type != ActionTypes.ACCEPT_REJECT:
            raise ValueError("Invalid action type: {}\nValid action types are: {}".format(action_type.name, ActionTypes.ACCEPT_REJECT.name))
        if not isinstance(action, Actions):
            raise ValueError("The action must be of type Actions ({})".format(type(action)))

        state = self.getState()
        assert self.getJobClass() is not None, "The job class of the arrived job is defined"

        # Convert the action to an ActionTypes class if given as integer
        if isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64):
            # Convert the action to an ActionTypes value
            action = Actions(action)

        if action not in [Actions.ACCEPT, Actions.REJECT]:
            raise ValueError("Invalid action: {}\nValid actions are: {}".format(action, Actions))

        if action == Actions.REJECT:
            # Reject the incoming job
            # => Just set the job class to None because no new job has yet arrived after taking the action
            # The state of the queue is left unchanged.
            self.setState((self.getQueueState(), None))
        elif action == Actions.ACCEPT:
            # => Accept the incoming job unless the system is at full capacity
            if self.getSystemOccupancy() == self.queue.getCapacity():
                # Set the job class to None because no new job has yet arrived at the moment of taking the action
                # The state of the queue is left unchanged.
                self.setState((self.getQueueState(), None))
                # Set the action to REJECT because the job was actually rejected, as the buffer was full!
                action = Actions.REJECT
            else:
                # Current state of the queue system
                queue_state = self.getQueueState()
                queue_next_state = list(copy.deepcopy(queue_state))
                queue_next_state[self.getJobClass()] += 1

                # Update the state of the system
                self.setState((tuple(queue_next_state), None))

        # Define the next state and the reward of going to that next state
        next_state = self.getState()
        # Reward observed by going from S(t) to S(t+1) via action A(t)
        if self.reward_func is not None:
            reward = self.reward_func(self, state, action, next_state, dict_params=self.dict_params_reward_func)
        else:
            reward = 0.0

        # Update the last action stored in the object
        self.setLastAction(action)

        return next_state, reward, {}

    #------ GETTERS ------#
    def getNumJobClasses(self):
        return self.queue.getNServers()

    def getJobClassRates(self):
        return list(self.queue.getBirthRates())

    def getSystemOccupancy(self):
        return super().getBufferSize()

    def getSystemOccupancyFromState(self, state :list):
        return super().getBufferSizeFromState(state)

    #------ SETTERS ------#
    def setState(self, state):
        """
        Sets the state ((j1, j2, ..., jc), i) of the queue environment from the number of jobs of each class and the arriving job class.
        - (j1, ... jc) is the number of jobs of each class being served by the system
        - i is the arriving job class

        Arguments:
        state: duple
            Duple containing the following information:
            - job class occupancy: list with the occupancy of each job class in the system
            - job_class: class of the arriving job
        """
        assert len(state) == 2
        # Extract the job class occupancy and the job class from the GIVEN state
        job_class_occupancy, job_class = self.getQueueStateFromState(state), self.getJobClassFromState(state)
        assert len(job_class_occupancy) == self.queue.getNServers()     # The number of servers is the capacity of the system in a Loss Network
        self.setQueueState(job_class_occupancy)
        self.job_class = job_class
        self.state = (self.getQueueState(), self.job_class)
