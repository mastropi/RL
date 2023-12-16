# -*- coding: utf-8 -*-
"""
Created on Mon Sep 4 17:09:10 2023

@author: Daniel Mastropietro
@description: Estimators of different types and purposes that cannot be clearly classed into any of the other estimators
defined in this directory.
"""

import warnings

import numpy as np

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.value_functions import StateValueFunctionApprox
from Python.lib.estimators import DEBUG_ESTIMATORS


class EstValueFunctionOfflineDeterministicNextState:
    """
    Offline estimator of the value function on a discrete-state / discrete-action environment where the next state
    given an action is deterministic.

    ASSUMPTIONS:
    - All possible actions are the same for each state and equal to the number of actions in the environment.
    - All possible actions are equally likely (random walk policy).

    The offline estimator consists in traversing all possible states and actions in the environment and updating
    the value function recursively using the Bellman equation on the state value function.

    This estimator is useful when there is no theoretical expression for the state value function.

    Arguments:
    env: EnvironmentDiscrete
        Discrete-state and discrete-action environment on which the state value function is estimated.

    gamma: float
        Discount parameter when learning the state value function. This is used in the Bellman equation.
    """
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        self.V = StateValueFunctionApprox(self.env.getNumStates(), self.env.getTerminalStates())

    def reset(self, reset_method, reset_params, reset_seed):
        self.env.reset()
        self.V.reset(method=reset_method, params_random=reset_params, seed=reset_seed)

    def estimate_state_values_random_walk(self, synchronous=True,
                                          max_delta=np.nan, max_delta_rel=1E-3, max_iter=1000, verbose=True, verbose_period=None,
                                          reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=1713):
        "Estimates the value function under the random walk policy"
        self.reset(reset_method, reset_params, reset_seed)

        if verbose_period is None:
            verbose_period = max_iter / 10

        print("\n\nTerminal states ({} out of {}): {}".format(len(self.env.getTerminalStates()), self.env.getNumStates(), self.env.getTerminalStates()))
        # WARNING: Only valid for MountainCarDiscrete environment
        #print("Positions: {}".format(self.env.get_positions()))
        #print("Velocities: {}".format(self.env.get_velocities()))
        # WARNING: Only valid for MountainCarDiscrete environment
        print("Initial V(s) estimate: {}".format(self.V.getValues()))

        max_deltaV_abs = np.Inf
        max_deltaV_rel_abs = np.Inf
        iter = 0
        while iter < max_iter and \
                (np.isnan(max_delta_rel) and max_deltaV_abs > max_delta or \
                 np.isnan(max_delta) and max_deltaV_rel_abs > max_delta_rel or \
                 not np.isnan(max_delta) and not np.isnan(max_delta_rel) and max_deltaV_abs > max_delta and max_deltaV_rel_abs > max_delta_rel):
            iter += 1
            values_prev = self.V.getValues()
            for s in self.env.getNonTerminalStates():
                #print("** state: {}".format(s))
                self.env.setState(s)
                n_actions_so_far = 0
                # Initialize the average observed value over all possible actions, whose value will be the updated V(s) value
                # once all actions have been taken
                V_mean_over_actions = 0.0
                for a in range(self.env.getNumActions()):
                    assert self.env.getState() == s, "getState(): {}, s: {}".format(self.env.getState(), s)
                    ns, reward, done, info = self.env.step(a)   # ns = next state

                    # NOTE 1: (2022/06/05) THIS ASSUMES THAT THE VALUE FUNCTION APPROXIMATION IS TABULAR!!
                    # In fact, I don't know how to write the Bellman equation in function approximation context
                    # (See chapter 9 in Sutton but I don't think it talks about this... it only talks about how to update
                    # the weights at each iteration, using Stochastic Gradient Descent (SGD).
                    # However, it does talk about the fixed point of this SGD algorithm, which is w = A^-1 * b
                    # where A and b are given in that chapter (pag. 206).)

                    # NOTE 2: At first we should perhaps update V(s) synchronosly, i.e. keep the same V(s) on the RHS
                    # until ALL states are updated.
                    # However, it seems that the asynchronous update done here works fine as well (recall class by Matt at UBA)
                    # and it even converges faster!
                    if DEBUG_ESTIMATORS:
                        print("state: {}, action: {} => next state = {}, reward = {} (done? {})".format(s, a, ns, reward, done))
                        # WARNING: Only valid for MountainCarDiscrete environment (because of call to self.env.get_state_from_index()
                        #print("state: {} ({}), action: {} => next state = {} ({}), reward = {} (done? {})" \
                        #      .format(s, self.env.get_state_from_index(s), a, ns, self.env.get_state_from_index(ns), reward, done))
                        # WARNING: Only valid for MountainCarDiscrete environment

                    # The new V(s) value is the PLAIN AVERAGE over all possible actions (since we are considering a random walk)
                    # If the policy were not random, we should multiply the V_observed value by the probability of taken action "a" given the current state s.
                    # This would require that this method receives e.g. a dictionary with each state as key and all the probabilities Pi(a|s) as values.
                    if synchronous:
                        # Use the state value computed at the PREVIOUS iteration as currently known value of the next state, V(ns)
                        V_observed = reward + self.gamma * values_prev[ns]
                    else:
                        # Use the CURRENT value of the next state, V(ns), even if it has been updated already in this iteration
                        # (i.e. without waiting for the value of all other states to be updated)
                        V_observed = reward + self.gamma * self.V.getValue(ns)
                    V_mean_over_actions = (n_actions_so_far * V_mean_over_actions + V_observed) / (n_actions_so_far + 1)

                    n_actions_so_far += 1

                    # Reset the state to the original state before the transition, so that the next action is applied to the same state
                    self.env.setState(s)
                # Update the state value of the currently analyzed state, V(s)
                self.V._setWeight(s, V_mean_over_actions)
                if DEBUG_ESTIMATORS:
                    if reward != 0:
                        print("--> New value for state s={} after taking all {} actions: {}".format(s, self.env.getNumStates(), self.V.getValue(s)))

            deltaV = (self.V.getValues() - values_prev)
            deltaV_rel = np.array([0.0  if dV == 0
                                        else dV / abs(V) if V != 0
                                                         else np.Inf
                                    for dV, V in zip(deltaV, values_prev)])
            mean_deltaV_abs = np.mean(np.abs(deltaV))
            max_deltaV_abs = np.max(np.abs(deltaV))
            max_deltaV_rel_abs = np.max(np.abs(deltaV_rel))
            if DEBUG_ESTIMATORS or verbose and (iter-1) % verbose_period == 0:
                print("Iteration {}: mean(|V_prev|) = {:.3f}, mean(|V|) = {:.3f}, mean|delta(V)| = {:.3f}, max|delta(V)| = {:.3f}, max|delta_rel(V)| = {:.7f}" \
                      .format(iter, np.mean(np.abs(values_prev)), np.mean(np.abs(self.V.getValues())), mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs))

        if  not np.isnan(max_delta) and max_deltaV_abs > max_delta or \
            not np.isnan(max_delta_rel) and max_deltaV_rel_abs > max_delta_rel or \
            not np.isnan(max_delta) and not np.isnan(max_delta_rel) and (max_deltaV_abs > max_delta or max_deltaV_rel_abs > max_delta_rel):
            warnings.warn("The estimation of the value function may not be accurate as the maximum relative absolute" \
                          " change in the last iteration #{} ({}) is larger than the maximum allowed ({})" \
                          .format(iter, max_deltaV_rel_abs, max_delta_rel))

        return iter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs

    def getV(self):
        return self.V

