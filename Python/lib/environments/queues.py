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
import copy
from enum import IntEnum, unique

import numpy as np

import gym
from gym import spaces

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../../setup.py')

from Python.lib.queues import GenericQueue, QueueMM

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
    OTHER = 99      # This is used for actions of ActionTypes = ASSIGN (where the action is actually the server number to which the accepted job is assigned)

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
    # The state is assumed to be a tuple (k, i) where k is the buffer size and i is the class of the arriving job
    if action == Actions.ACCEPT:
        job_class = state[1]
        return env.getRewardForJobClassAcceptance(job_class)
    else:
        return 0.0

def rewardOnJobRejection_ExponentialCost(env, state, action, next_state, dict_params=None):
    if dict_params is None:
        dict_params = dict({'buffer_size_ref': COST_EXP_BUFFER_SIZE_REF})
    if 'buffer_size_ref' not in dict_params.keys():
        raise ValueError("Parameter 'buffer_size_ref' must be given in the parameters dictionary containing"
                         " the reference buffer size, above which there is an exponentially increasing cost with"
                         " the buffer size when a job is rejected.")

    "Negative reward received when an incoming job is rejected as a function of the queue's buffer size associated to the given state"
    if action == Actions.REJECT:
        # Blocking occurred
        # => There is a negative reward or cost
        assert env.getBufferSizeFromState(state) == env.getBufferSizeFromState(next_state), \
            "At REJECT, the queue's buffer size after rejection ({}) is the same as before rejection ({})" \
            .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))
        reward = -costBlockingExponential(env.getBufferSizeFromState(state), dict_params['buffer_size_ref'])
        assert reward < 0.0, "The reward of blocking is negative ({})".format(reward)
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
    B = 1       # Blocking associated just to the fact the queue is blocked
    b = 3.0     # Base of the exponential function
    #cost = B if buffer_size <= buffer_size_ref else B * b**(buffer_size - buffer_size_ref)
    cost = B * (1 + b**(buffer_size - buffer_size_ref))
    if False:
        print("Computing cost for buffer size = {} with sref = {} --> Cost = {:.3f}".format(buffer_size, buffer_size_ref, cost))
    if False and cost != B:
        print("The cost is in the exponential part (increasing with `buffer size - REF buffer size`): buffer={}, ref={}".format(buffer_size, buffer_size_ref))
    return cost
#---------------------------- Reward functions that can be used in different environments -----------------------------#


class EnvQueueSingleBufferWithJobClasses(gym.Env):
    """
    Queue environment with a single buffer receiving jobs of different classes
    being served by one or multiple servers.

    The possible actions for an agent interacting with the queue are:
    - Accept or reject an incoming job of a given class --> action_type = ActionTypes.ACCEPT_REJECT
    - Choose the server to which the accepted incoming job is assigned --> action_type = ActionTypes.ASSIGN

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

    dict_params_reward_funct: (opt) dict
        Dictionary containing non-standard parameters (i.e. besides the state, the action and the next state)
        on which the reward function `reward_func` depends on.
        Ex: dict_params_reward_func = {'buffer_size_ref': 20}
        default: None, in which case the default parameter values are used
    """

    def __init__(self, queue: QueueMM, job_class_rates: list,
                 reward_func, rewards_accept_by_job_class: list,
                 dict_params_reward_func: dict=None):
        #-- Environment
        self.queue = queue
        self.job_class = None
        self.job_class_rates = job_class_rates

        #-- Rewards
        self.reward_func = reward_func
        self.dict_params_reward_func = dict_params_reward_func
        self.rewards_accept_by_job_class = rewards_accept_by_job_class

        # Create temporary variables that are shorter to reference and still easy to understand
        K = self.queue.getCapacity()
        J = len(self.job_class_rates)

        # Action and observation spaces
        # TODO: (2021/10/22) Make sure we use these action and observation spaces in the implementation of actions and spaces in the environment
        # It seems to me that currently this information is never used.
        #self.action_space = spaces.Discrete(2)  # Either Reject (0) or Accept (1) an arriving job class
        self.action_space = Actions
        self.observation_space = spaces.Tuple((spaces.Discrete(K+1), spaces.Discrete(J)))

        # Last action taken by the agent interacting with the environment
        # (this piece of information is needed e.g. when learning the policy of the agent interacting
        # with the environment. See e.g. LeaPolicyGradient.learn())
        self.action = None

        # Seed used in the generation of random numbers during the interaction with the environment.
        # For instance to decide on blocking in a parameterized policy where theta defines a state with
        # RANDOM policy of acceptance (see e.g. GenericParameterizedPolicyTwoActions.choose_action())
        self.seed = None

        start_state = ([0]*self.queue.getNServers(), self.job_class)
        self.reset(start_state)

    def reset(self, state=None):
        """
        Resets the state of the environment and the last action it received

        Arguments:
        state: (opt) duple
            Duple containing the following information:
            - server sizes: list with the size of each queue in each server making up the queue system
            - job_class: class of the arriving job
            default: None, in which case the queue is reset to the state with all 0s in the servers and None job class
        """
        if state is None:
            state = ([0] * self.queue.getNServers(), None)
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
                raise ValueError(
                    "Invalid action: {}\nValid actions are: {}".format(action, Actions))

            if action == Actions.REJECT:
                # Reject the incoming job
                # => Just set the job class to None because no new job has yet arrived after taking the action
                self.setState((self.queue.getServerSizes(), None))
            elif action == Actions.ACCEPT:
                # => Accept the incoming job unless the queue's buffer is at full capacity
                # Note that when the job is accepted NOTHING is done here, because at this point
                # we don't know how to assign the accepted to a server in the queue, as for that we need the
                # assignment policy, and this is a DIFFERENT policy, i.e. with actions of a different type
                # (the ActionsType.ASSIGN type).
                # So, accepting the job means simply "not rejecting it", and actually the ACTUAL acceptance
                # (i.e. the assignment of the job to a server) will be done separately when calling this step() function
                # with an ActionsType.ASSIGN type of action.
                if self.getBufferSize() == self.queue.getCapacity():
                    # Set the job class to None because no new job has yet arrived at the moment of taking the action
                    self.setState((self.queue.getServerSizes(), None))
                    # Set the action to REJECT because the job was actually rejected, as the buffer was full!
                    action = Actions.REJECT
        elif action_type == ActionTypes.ASSIGN:
            # Current state of the queue system
            queue_state = self.getQueueState()
            # Define the next state of the queue system based on the action
            # (which represents the server to which an incoming job is assigned)
            queue_next_state = copy.deepcopy(queue_state)
            queue_next_state[action] += 1

            # Change the state of the system
            self.setState((queue_next_state, None))

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
    def get_seed(self):
        return self.getSeed()
    def getSeed(self):
        return self.seed

    def getQueue(self):
        return self.queue

    def getCapacity(self):
        return self.queue.getCapacity()

    def getJobClassRates(self):
        return self.job_class_rates

    def getServiceRates(self):
        return self.queue.getDeathRates()

    def getNumJobClasses(self):
        return len(self.job_class_rates)

    def getNumServers(self):
        return self.queue.getNServers()

    def getQueueState(self):
        return self.queue.getServerSizes()

    def getState(self):
        return (self.getBufferSize(), self.getJobClass())

    def getBufferSize(self):
        return self.queue.getBufferSize()

    def getBufferSizeFromState(self, state :list):
        queue_state = state[0]
        return np.sum(queue_state)

    def getJobClass(self):
        return self.job_class

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
    def setParamsRewardFunc(self, dict_params):
        self.dict_params_reward_func = dict_params

    def _setServerSizes(self, sizes : int or list or np.array):
        "Sets the size of the servers in the queue"
        self.queue.setServerSizes(sizes)

    def setJobClass(self, job_class: int or None):
        "Sets the job class of a new arriving job. The job class can be None."
        if not (
                job_class is None or \
                int(job_class) == job_class and 0 <= job_class <= self.getNumJobClasses()
                ):
            raise ValueError(
                "The job class is invalid ({}). It must be an integer between {} and {}".format(job_class, 0, self.getNumJobClasses() - 1))
        self.job_class = job_class

    def setState(self, state):
        """
        Sets the state (k, i) of the queue environment from the server sizes and the arriving job class.
        - k is the buffer size
        - i is the arriving job class

        Arguments:
        state: duple
            Duple containing the following information:
            - server sizes: list with the size of each queue in each server making up the queue system
            - job_class: class of the arriving job
        """
        assert len(state) == 2
        sizes = state[0]
        job_class = state[1]
        self.queue.setServerSizes(sizes)
        self.job_class = job_class

    def setLastAction(self, action):
        self.action = action

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

    # -- Equivalent job arrival rates (by server)
    print("Job arrival rates: {}".format(env.job_class_rates))

    # -- Arriving job on buffer not full
    buffer_size_orig = 0
    job_class = 1
    env.setJobClass(job_class)
    state = env.getState()
    assert state == (0, job_class)
    # Action = reject
    next_state, reward, info = env.step(Actions.REJECT, ActionTypes.ACCEPT_REJECT)
    assert next_state == (0, None)
    assert reward == 0
    # Action = accept
    env.setJobClass(job_class)
    next_state, reward, info = env.step(Actions.ACCEPT, ActionTypes.ACCEPT_REJECT)
    assert next_state == (buffer_size_orig, job_class)
    assert reward == env.getRewardForJobClassAcceptance(job_class)
    # Now we assign the job to the server
    next_state, reward, info = env.step(0, ActionTypes.ASSIGN)
    assert next_state == (buffer_size_orig + 1, None)
    assert reward == 0

    # -- Arriving job on full buffer
    job_class = 1
    env.setState((env.getCapacity(), job_class))
    state = env.getState()
    assert state == (env.getCapacity(), job_class)
    # Action = reject
    next_state, reward, info = env.step(Actions.REJECT, ActionTypes.ACCEPT_REJECT)
    assert next_state == (env.getCapacity(), None)
    assert reward == 0
    # Action = accept (not taken because buffer is full)
    env.setJobClass(job_class)
    next_state, reward, info = env.step(Actions.ACCEPT, ActionTypes.ACCEPT_REJECT)
    assert next_state == (env.getCapacity(), None)
    assert reward == 0
