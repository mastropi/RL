# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:26:20 2020

@author: Daniel Mastropietro
@description: Reinforcement Learning simulators are defined on a given environment and agent
interacting with the environment.
The environments are assumed to have the following methods defined:
    - reset()
    - step(action) --> the method that takes the given action and updates the state, returning a tuple:
        (next_state, reward, done, info)
and getters:
    - getNumStates()
    - getNumActions()
    - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
        (responsible for defining the initial state when running the simulation)
    - getState() --> returns the current state of the environment
and setters:
    - setInitialStateDistribution()

The environments of class gym.toy_text.Env satisfy the above conditions.
"""

import numpy as np

class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a given environment `env`
    and an `agent`.
    """

    def __init__(self, env, agent, debug=False):
        self.env = env
        self.agent = agent
        self.debug = debug
        self._isd_orig = None   # Copy of Initial State Distribution of environment in case we need to change it

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the initial state distribution array in case it has been possibly changed
        # (e.g. when we want to define a specific initial state)
        if self._isd_orig:
            self.setInitialStateDistribution(self._isd_orig)

    def play(self, start=None, nrounds=100):
        # Define initial state
        if start:
            nS = self.env.getNumStates()
            if start >= nS:
                raise Warning('start parameter must be between 0 and {}.' \
                              'A start value will be selected RANDOMLY.'.format(nS-1))
            else:
                # Change the initial state distribution of the environment so that
                # the environment resets to start at the given 'start' state.
                self._isd_orig = self.env.getInitialStateDistribution()
                isd = np.zeros(nS)
                isd[start] = 1.0
                self.env.setInitialStateDistribution(isd)

        for _ in range(nrounds):
            learner = self.agent.getLearner()

            self.env.reset()
            done = False
            if self.debug:
                print("\nStarts at state {}".format(self.env.getState()))

            # Time step in the episode (the first time step is t = 0
            t = -1
            while not done:
                t += 1
                
                # Current state and action on that state leading to the next state
                state = self.env.getState()
                action = self.agent.getPolicy().choose_action()
                next_state, reward, done, info = self.env.step(action)
                if self.debug:
                    print("| t: {} ({}) -> {}".format(t, action, next_state), end=" ")

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                learner.learn_pred_V(t, state, action, next_state, reward, done, info)
                #if self.debug:
                #    print("time step {}: udpated Z: {}".format(t, learner.getZ()))

                #if self.debug:
                #    print("-> {}".format(self.env.getState()), end=" ")

            if self.debug:
                print("--> Done [{} iterations] at state {} with reward {}".
                      format(t+1, self.env.getState(), reward))

        # TODO: DO WE NEED THIS RESTORE OF THE INITIAL STATE DISTRIBUTION OR THE __exit__() METHOD SUFFICES??
#        if self.isd_orig:
#            self.env.setInitialStateDistribution(self.isd_orig)
