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
else:
    from . import EnvironmentDiscrete

class deprecated_EnvQueueSingleBufferWithJobClasses(EnvironmentDiscrete):
    """
    Queue environment with a single buffer receiving jobs of different classes.

    Arguments:
    capacity: positive int
        Buffer capacity.

    num_job_classes: positive int
        Number of different job classes that can arrive to the queue.

    rewards: list
        List of rewards for each job class.
        Its length must be equal to the number of job classes.
    """

    def __init__(self, capacity :int, num_job_classes :int, rewards :list):
        if len(rewards) != num_job_classes:
            raise ValueError("The number of rewards ({}) must be the same as the number of job classes ({})".format(len(rewards), num_job_classes))

        self.capacity = capacity
        self.num_job_classes = num_job_classes
        self.rewards = rewards

        # Create temporary variables that are shorter to reference and still easy to understand
        K = self.capacity
        J = self.num_job_classes

        # STATES: Each state is a pair (k,j) where k is the buffer occupancy (0 ... K) and j is the arriving job class (1 ... J)
        nS = (K+1)*J

        # ACTIONS for each STATE: Either accept or reject an arriving job at a given state (k,j)
        nA = 2
        P = {}
        states = np.arange(nS).reshape([K+1, J])
        it = np.nditer(states, flags=['multi_index'])

        # Initial state distribution is 0 for k > 0 with uniform distribution on the arriving job classes
        # Note that we need to store this information as a 1D array because this is how it is required
        # by the constructor of DiscreteEnv in open gym.
        isd = np.zeros(nS)
        while not it.finished:
            s = it.iterindex
            k, j = it.multi_index

            # The states with buffer size = 0 are the only possible initial states
            if k == 0:
                isd[s] = 1.0 / J

            # Transition matrix and rewards: P[s][a] = (prob, next_state, reward, is_terminal)
            # Given that we are state 's' and take action 'a':
            # prob: probability of going to state 'next_state'
            # next_state: a possible next_state when taking action 'a'
            # reward: reward received when going to state 'next_state'
            # is_terminal: whether 'next_state' is a terminal state
            if k < K:
                P[s] = {a: [] for a in range(nA)}
                # NOTE: Once an action is taken, with no further information,
                # the next state is NOT uniquely defined as it depends on the class of the NEXT arriving job.
                # Therefore we can assume here that the next arriving job is random but we can
                # update the next state from the program that controls the environment evolution
                # by imposing the actual next arriving job. 
                # - If the action taken is 0 (reject) the possible next states are (k, 0), ..., (k, J-1) 
                # - If the action taken is 1 (accept) the possible next states are (k+1, 0), ..., (k+1, J-1) 
                P[s][0] = [(1.0/J, k*J + j, 0.0, False) for j in range(J)]
                P[s][1] = [(1.0/J, (k+1)*J + j, rewards[j], False) for j in range(J)]
            else:
                # The buffer is full
                # => The only action is to reject the arriving job
                P[s] = {0: [(1.0/J, k*J + j, 0.0, False) for j in range(J)]}

            it.iternext()

        super(deprecated_EnvQueueSingleBufferWithJobClasses, self).__init__(nS, nA, P, isd, dim=2)

    def getCapacity(self):
        return self.capacity

    def getNumJobClasses(self):
        return self.num_job_classes

    def getRewards(self):
        return self.rewards
    
    
class EnvQueueSingleBufferWithJobClasses(gym.Env):
    """
    Queue environment with a single buffer receiving jobs of different classes.

    Arguments:
    capacity: positive int
        Buffer capacity.

    num_job_classes: positive int
        Number of different job classes that can arrive to the queue.

    rewards: list
        List of rewards for each job class.
        Its length must be equal to the number of job classes.
    """

    def __init__(self, capacity :int, num_job_classes :int, rewards :list, seed=None):
        if len(rewards) != num_job_classes:
            raise ValueError("The number of rewards ({}) must be the same as the number of job classes ({})".format(len(rewards), num_job_classes))

        self.capacity = capacity
        self.num_job_classes = num_job_classes
        self.rewards = rewards
        self.seed = seed

        # Create temporary variables that are shorter to reference and still easy to understand
        K = self.capacity
        J = self.num_job_classes

        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # Either Reject (0) or Accept (1) an arriving job class
        self.observation_space = spaces.Tuple( (spaces.Discrete(K+1), spaces.Discrete(J)) )

        self.reset()

    def reset(self):
        self.buffer_size = 0
        self.job_class = None   # np.random.choice( self.action_space.n )
        # Reset the seed
        np.random.seed(self.seed)

    def np_random(self):
        return np.random()

    def step(self, action):
        assert self.job_class is not None, "The job class of the arrived job is defined"

        if not self.action_space.contains(action):
            raise ValueError("Invalid action: {}\nValid actions are: 0, 1, ..., {}".format(action, self.action_space.n-1))

        reward = 0
        if action == 1:
            if self.buffer_size == self.capacity:
                warn("Action '{}' not taken because the buffer is at full capacity ({})!".format(action, self.capacity))
            else:
                reward = self.rewards[ self.job_class ]
                # Change the state of the system
                # (the job class is set to None because no new job has yet arrived)
                self.buffer_size += 1
        
        # Reset the job class to None because it was just considered for the given action
        self.job_class = None

        return self.getState(), reward, False, {}        

    #------ GETTERS ------#
    def getCapacity(self):
        return self.capacity

    def getNumJobClasses(self):
        return self.num_job_classes

    def getRewards(self):
        return self.rewards

    def getActions(self):
        return range(self.action_space.n)

    def getState(self):
        return (self.buffer_size, self.job_class)

    #------ SETTERS ------#
    def setBufferSize(self, buffer_size :int):
        "Sets the buffer size of the queue"
        if int(buffer_size) != buffer_size or buffer_size < 0 or buffer_size > self.capacity:
            raise ValueError("The buffer size value is invalid ({}). It must be an integer between {} and {}".format(buffer_size, 0, self.capacity))
        self.buffer_size = buffer_size

    def setJobClass(self, job_class):
        "Sets the job class of a new arriving job"
        if int(job_class) != job_class or job_class < 0 or job_class >= self.num_job_classes:
            raise ValueError("The job class is invalid ({}). It must be an integer between {} and {}".format(job_class, 0, self.num_job_classes-1))
        self.job_class = job_class


if __name__ == "__main__":
    #------------------------- Unit tests -----------------------------#
    #--- Test #1: Create a queue environment in DEPRECATED class
    print("Running test #1: Deprecated queue environment inheriting from toy_text.discrete.DiscreteEnv...")
    env = deprecated_EnvQueueSingleBufferWithJobClasses(6, 4, [1, 0.8, 0.3, 0.2])
    K = env.getCapacity()
    J = env.getNumJobClasses()
    R = env.getRewards()
    nS = env.getNumStates()

    print("Initial state distribution:")
    s = -1
    for k in range(K):
        print("k={}:".format(k), end=" [")
        sep = ""
        for j in range(J):
            s += 1
            print("{}{}".format(sep, env.isd[s]), end="")
            sep = ", "
        print("]")

    print("Transition probabilities:")
    s = -1
    for k in range(K):
        for j in range(J):
            s += 1
            print("s=({}, {}):".format(k, j))
            for a in sorted(env.P[s].keys()):
                print("a={}:".format(a), end= "\n")
                for transition_info in env.P[s][a]:
                    print("\tnext state: {} (p={})".format(transition_info[1], transition_info[0]))
            print("")

    #--- Test #2: Create a queue environment
    print("Running test #2: Queue environment inheriting from gym.Env...")
    env = EnvQueueSingleBufferWithJobClasses(6, 4, [1, 0.8, 0.3, 0.2])
    K = env.getCapacity()
    J = env.getNumJobClasses()
    R = env.getRewards()
    nS = len(env.getActions())

    #-- Arriving job on buffer not full
    buffer_size_orig = 0
    job_class = 1
    env.setJobClass(job_class)
    state = env.getState()
    assert state == (0, job_class)
    # Action = reject
    next_state, reward, done, info = env.step(0)
    assert next_state == state
    assert reward == 0
    # Action = accept
    next_state, reward, done, info = env.step(1)
    assert next_state == (buffer_size_orig + 1, None)
    assert reward == env.getRewards()[job_class]

    #-- Arriving job on full buffer
    env.setBufferSize(env.capacity)
    job_class = 1
    env.setJobClass(job_class)
    state = env.getState()
    assert state == (env.capacity, job_class)
    # Action = reject
    next_state, reward, done, info = env.step(0)
    assert next_state == state
    assert reward == 0
    # Action = accept (not taken because buffer is full)
    next_state, reward, done, info = env.step(1)
    assert next_state == state
    assert reward == 0
