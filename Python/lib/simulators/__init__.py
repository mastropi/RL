# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:23:07 2022

@author: Daniel Mastropietro
@description: Definition of functions normally used in simulations. Ex: step(), check_done()
"""

from enum import Enum, unique

import numpy as np

from Python.lib.agents import GenericAgent
from Python.lib.agents.policies import PolicyTypes


@unique
class LearningMethod(Enum):
    MC = 1
    FV = 2


def check_done(tmax, t, state, action, reward):
    """
    Checks whether the simulation is done

    tmax: int
        Maximum discrete time allowed for the simulation.

    t: int
        Current queue simulation time.

    state: Environment dependent
        S(t): state of the environment at time t, BEFORE the action is taken.

    action: Environment dependent
        A(t): action received by the environment at time t.

    reward: float
        R(t+1): reward yielded by the environment after taking action A(t) at state S(t).

    Return: bool
        Whether the queue simulation is done because the maximum number of iterations has been reached.
    """
    if t < tmax:
        done = False
    else:
        done = True

    return done


def step(t, env, agent: GenericAgent, policy_type: PolicyTypes):
    """
    The given agent performs an action on the given environment out of the possible actions defined in the given policy

    Arguments:
    t: int
        Current queue transition associated to the current step.

    env: environment
        The environment where the agent acts.

    agent: Agent
        The agent interacting with the environment.
        It should have the act() method defined.

    policy_type: PolicyTypes
        Type of policy to apply when stepping. Possible values are:
        - PolicyTypes.ACCEPT
        - PolicyTypes.ASSIGN

    Return: tuple
    Tuple containing the following elements:
    - action: the action taken by the agent on the given policy
    - observation: the next state on which the queue transitions to after the action taken
    - reward: the reward received by the agent after taking the action and transitioning to the next state
    - info: dictionary with relevant additional information
    """
    action, observation, reward, info = agent.act(env, policy_type)

    return action, observation, reward, info


def update_trajectory(agent, t_total, t_sim, state, action, reward):
    """
    Updates the trajectory stored in the state value function learner and in the policy learner of the given agent

    Arguments:
    agent: Agent
        Agent that is responsible of performing the actions on the environment and learning from them
        on which the trajectory to update is stored.

    t_total: int
        Total time, used to store the trajectory for the policy learning.
        Normally this would be the total simulation time computed from the learning steps.

    t_sim: float
        Continuous simulation time, used to store the trajectory for the value function learning.

    state: State (however the state is defined, e.g. as the buffer size)
        State at the given time to store in the trajectory.

    action: Action
        Action taken at the given time to store in the trajectory.

    reward: float
        Reward received by the agent after taking the given action at the given state, to store in the trajectory.
    """
    if agent.getLearnerV() is not None:
        agent.getLearnerV().update_trajectory(t_sim, state, action, reward)
    if agent.getLearnerP() is not None:
        # DM-2021/11/28: This assertion is no longer true because we are now storing in the trajectory ALSO the states
        # occurring just before the FIRST DEATH event happening after a BIRTH event (so that we can show it in the
        # trajectory plot we show at the end of the simulation and thus avoid the suspicion that something is wrong
        # when we observe no change in the buffer size from one time step to the next --given that a new time step
        # is created ONLY when a new job arrives (i.e. a BIRTH event occurs)
        # assert action is not None, "The action is not None when learning the policy"
        agent.getLearnerP().update_trajectory(t_total, state, action, reward)


def show_messages(verbose, verbose_period, t_learn):
    return verbose and np.mod(t_learn - 1, verbose_period) == 0

