# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:15:04 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all learners.

All learner classes defined in the current package should:
a) Implement the following attributes:
    - env --> the environment on which the learning takes place
    - V --> an object containing the information on how the state value function is ESTIMATED.
    - Q --> an object containing the information on how the state-action value function is ESTIMATED.
    - alpha --> the learning rate
    - gamma --> the reward discount parameter
b) Implement the following methods:
All learners are assumed to have the following methods defined:
    - reset() --> resets the state of the learner to start learning anew
        (e.g. all the estimates of the value functions are reset to 0)
    - setParams() --> set the parameters of the learner (if any)
    - learn_pred_V() --> prediction problem: learns the state value function under the current policy (V(s))
    - learn_pred_Q() --> prediction problem: learns the action-value function under the currenty policy (Q(s,a))
    - learn_ctrl_policy() --> control problem: learns the optimal policy

VALUE FUNCTIONS:
The value functions used by the learner classes are assumed to:
a) Be defined in terms of weights, e.g. the state value function for state s is V(s,w),
where w is the vector of weights applied to a set of features X.
Note that this assumption does NOT offer any limitation, since a tabular value function
can be defined using binary/dummy features. 

b) Have the following methods defined:
- reset(): resets the vector w of weigths to their initial estimates 
- getWeights(): reads the vector w of weights
- setWeights(): updates the vector w of weights
- getValue(): reads the value function for a particular state or state-action
- getValues(): reads the value function for ALL states or state-actions
"""

import numpy as np


class Learner:
    """
    Class defining methods that are generic to ALL environments.
    
    NOTE: Before using any learner the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: one when the learner is constructed
    and one prior to the first simulation, making e.g. the episode method be equal to 2 at
    the first simulation as opposed to the correct value 1.
    """

    def __init__(self):
        # Information of the observed trajectory at the END of the episode
        # (so that it can be retrieved by the user if needed as a piece of information)
        self.states = []
        self.rewards = []

        # Episode counter
        self.episode = 0

    def reset(self, reset_episode=False, reset_value_functions=False):
        """
        Resets the variables that store information about the episode
        
        Parameters:
        reset_episode: bool, optional
            Whether to reset the episode to the first one.

        reset_value_functions: bool, optional
            Whether to reset all the value functions to their initial estimates as well. 
        """
        if reset_episode:
            self.episode = 0
        
        # Increase episode counter
        # (note that the very first episode run is #1 because reset() is called by __init__()) 
        self.episode += 1
        # Reset the attributes that keep track of states and rewards received during learning
        self._reset()

        # Only reset the initial estimates of the value functions at the very first episode
        # (since each episode should leverage what the agent learned so far!)
        if self.episode == 1 or reset_value_functions:
            self.V.reset()

    def _reset(self):
        """
        Resets internal structures used during learning
        (all attributes reset here should start with an underscore, i.e. they should be private)
        """

        # Store the _states visited in the episode and their count
        self._states = []
        self._state_counts = np.zeros(self.env.getNumStates())

        # Store the _rewards obtained after each action
        # We initialize the _rewards with one element equal to 0 for ease of notation
        # when retrieving the state and the reward received at each time t
        # because the state is defined for t = 0, ..., T-1 
        # while the reward is defined for t = 1, ..., T
        self._rewards = [0]

    def _update_trajectory(self, state, reward):
        "Updates the trajectory based on the current state and the observed reward"
        self._states += [state]
        self._rewards += [reward]

    def store_trajectory(self):
        "Stores the trajectory observed during the episode"
        # Remove any trajectory stored from a previous run from memory
        del self.states
        del self.rewards
        # Assign the new trajectory observed in the current episode 
        self.states = self._states.copy()
        self.rewards = self._rewards.copy()

    def getV(self):
        "Returns the object containing information about the state value function estimation"
        return self.V

    def getQ(self):
        "Returns the object containing information about the state-action value function estimation"
        return self.Q
