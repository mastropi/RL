# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:23:07 2022

@author: Daniel Mastropietro
@description: Definition of functions normally used in simulations. Ex: step(), check_done()
"""

from enum import Enum, unique

import numpy as np

from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses
from Python.lib.queues import QueueMM

from Python.lib.utils.computing import compute_job_rates_by_server

@unique
class LearningMethod(Enum):
    MC = 1
    FV = 2


def define_queue_environment_and_agent(dict_params: dict):
    """
    Define the queue environment and agent that will be used in the simulation

    Arguments:
    dict_params: dict
        Dictionary defining the simulation parameters with at least the following entries:
        - 'environment': parameters for the queue environment.
        - 'policy': parameters for the policy.
        - 'learners': parameter for the value functions and policy learners.
        - 'agent': parameters for the agent.

        Each entry in turn, should contain at least the following entries:
        - 'environment': dictionary defining the queue environment parameters with the following entries:
            - 'capacity': capacity K of the queue's buffer.
            - 'nservers': number of servers c in the queue system.
            - 'job_class_rates': list containing the Job arrival rates by job class.
            - 'service_rates': list containing the service rates of the servers in the queue system.
            - 'policy_assignment_probabilities': list of lists defining the job class to server assignment probabilities.
                Each inner list should have as many elements as servers in the system.
                The outer list should have as many elements as job classes accepted by the system.
                Ex: [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] when nservers = 3 and len(job_class_rates) = 2
            - 'reward_func': function returning the reward given by the environment for each state and action.
            - 'rewards_accept_by_job_class': list of rewards for the acceptance of the different job classes
            whose rate is given in job_class_rates.
        - 'policy': dictionary defining the type of parameterized policy to use when learning the policy with the following entries:
            - 'parameterized_policy': object defining the parameterized policy.
            - 'theta': the initial theta parameter value for the parameteried policy.
        - 'learners': dictionary defining the learners of value functions and policies with the following entries:
            - 'V': dictionary defining the state value function parameters with the following entries:
                - 'learner': learner for the state value function V.
                - 'params': dictionary with the learner parameters as follows:
                    - 'gamma': discount factor used for the return G(t).
            - 'Q': dictionary defining the action-state value function parameters with the following entries:
                NOT USED FOR NOW.
                - 'learner': learner for the action-state value function V.
                - 'params': dictionary with the learner parameters. Undefined for now.
            - 'P': dictionary defining the policy learner parameters with the following entries:
                - 'learner': learner of the policy P.
                - 'params': dictionary with the learner parameters:
                    - 'alpha_start': initial value for the learning rate alpha.
                    - 'adjust_alpha': whether to adjust the learning rate alpha as learning progresses.
                    - 'func_adjust_alpha': function used to adjust the learning rate alpha.
                    - 'min_time_to_update_alpha': minimum time step at which alpha is updated/adjusted.
                    - 'alpha_min': minimum learning rate allowed during the learning process.
                    - 'fixed_window': whether to use a fixed window in the estimation of the observed return (I think).
                    - 'clipping': whether to use clipping when updating the policy parameter theta.
                    - 'clipping_value': value at which the delta(theta) is clipped every time it is updated (e.g. +1.0).
        - 'agent': dictionary with the agent setup, with the following entries:
            - 'agent': agent object that interacts with the environment.

    Return: Tuple
    Triple containing:
    - the queue environment created from the input parameters
    - the loads of each server in the queue system, computed from the job class arrival rates
    and the given assignment probabilities.
    - the agent interacting with the environment, with its policies and learners.
    """
    set_entries_params = {'environment', 'policy', 'learners', 'agent'}
    if not set_entries_params.issubset(dict_params.keys()):
        raise ValueError("Missing entries in the dict_params dictionary: {}" \
                         .format(set_entries_params.difference(dict_params.keys())))

    # Environment
    set_entries_params_environment = {'capacity', 'nservers', 'job_class_rates', 'service_rates',
                                      'policy_assignment_probabilities', 'reward_func',
                                      'rewards_accept_by_job_class'}
    if not set_entries_params_environment.issubset(dict_params['environment'].keys()):
        raise ValueError("Missing entries in the dict_params['environment'] dictionary: {}" \
                         .format(set_entries_params_environment.difference(dict_params['environment'].keys())))

    # Policy (Job Acceptance)
    set_entries_params_policy = {'parameterized_policy', 'theta'}
    if not set_entries_params_policy.issubset(dict_params['policy'].keys()):
        raise ValueError("Missing entries in the dict_params['policy'] dictionary: {}" \
                         .format(set_entries_params_policy.difference(dict_params['policy'].keys())))

    # Learners (of V, Q, and P)
    set_entries_params_learners = {'V', 'Q', 'P'}
    if not set_entries_params_learners.issubset(dict_params['learners'].keys()):
        raise ValueError("Missing entries in the dict_params['learners'] dictionary: {}" \
                         .format(set_entries_params_learners.difference(dict_params['learners'].keys())))

    # Agent
    set_entries_params_agent = {'agent'}
    if not set_entries_params_agent.issubset(dict_params['agent'].keys()):
        raise ValueError("Missing entries in the dict_params['agent'] dictionary: {}" \
                         .format(set_entries_params_agent.difference(dict_params['agent'].keys())))

    # Store the parameters into different dictionaries, for easier handling below
    dict_params_env = dict_params['environment']
    dict_params_policy = dict_params['policy']
    dict_params_learners = dict_params['learners']
    dict_params_agent = dict_params['agent']

    # Queue
    policy_assign = PolJobAssignmentProbabilistic(dict_params_env['policy_assignment_probabilities'])
    job_rates_by_server = compute_job_rates_by_server(dict_params_env['job_class_rates'],
                                                      dict_params_env['nservers'],
                                                      policy_assign.getProbabilisticMap())
    # Queue M/M/c/K
    queue = QueueMM(job_rates_by_server, dict_params_env['service_rates'],
                    dict_params_env['nservers'], dict_params_env['capacity'])

    # Queue environment
    env_queue = EnvQueueSingleBufferWithJobClasses(queue, dict_params_env['job_class_rates'],
                                                      dict_params_env['reward_func'],
                                                      dict_params_env['rewards_accept_by_job_class'])
    rhos = [l / m for l, m in zip(job_rates_by_server, dict_params_env['service_rates'])]

    # Acceptance and assignment policies
    policies = dict({PolicyTypes.ACCEPT: dict_params_policy['parameterized_policy'](env_queue,
                                                                                    theta=dict_params_policy['theta']),
                     PolicyTypes.ASSIGN: policy_assign})

    # Learners (for V, Q, and P)
    learnerV = dict_params_learners['V']['learner'](env_queue,
                                                    gamma=dict_params_learners['V']['params'].get('gamma'))
    learnerQ = None
    if dict_params_learners['P']['learner'] is None:
        learnerP = None
    else:
        learnerP = dict_params_learners['P']['learner'](env_queue, policies[PolicyTypes.ACCEPT],
                                                        learnerV,
                                                        alpha=dict_params_learners['P']['params'].get('alpha_start'),
                                                        adjust_alpha=dict_params_learners['P']['params'].get('adjust_alpha'),
                                                        func_adjust_alpha=dict_params_learners['P']['params'].get('func_adjust_alpha'),
                                                        min_time_to_update_alpha=dict_params_learners['P']['params'].get('min_time_to_update_alpha'),
                                                        alpha_min=dict_params_learners['P']['params'].get('alpha_min'),
                                                        fixed_window=dict_params_learners['P']['params'].get('fixed_window'),
                                                        clipping=dict_params_learners['P']['params'].get('clipping'),
                                                        clipping_value=dict_params_learners['P']['params'].get('clipping_value'))

    # TODO: (2022/01/17) Can we add the following assertion at some point?
    # from Python.lib.agents.learners import GenericLearner
    # assert isinstance(learnerV, GenericLearner)
    learners = dict({LearnerTypes.V: learnerV,
                     LearnerTypes.Q: learnerQ,
                     LearnerTypes.P: learnerP
                     })

    # Agent operating on the given policies and learners
    agent = dict_params_agent['agent'](env_queue, policies, learners)

    return env_queue, rhos, agent

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

