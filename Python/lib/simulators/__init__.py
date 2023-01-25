# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:23:07 2022

@author: Daniel Mastropietro
@description: Definition of functions normally used in simulations. Ex: step(), check_done()
"""

from enum import Enum, unique
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses
from Python.lib.queues import QueueMM

from Python.lib.utils.basic import is_scalar, is_integer
from Python.lib.utils.computing import compute_job_rates_by_server

@unique
class LearningMethod(Enum):
    MC = 1
    FV = 2


class SetOfStates:
    """
    Class that can be used to define a set of states in a compact way

    Either one of the two parameters must not be None.
    If both are given (i.e. not None), preference is given to parameter `states`, which then is the one defining
    the set of states belonging to the set.

    Arguments:
    states: (opt) set of tuples
        The set of states belonging to the set.
        Ex: {(3, 2, 1), (4, 2, 5)}
        default: None

    state_boundaries: (opt) tuple
        A list defining the state value in each dimension of the state vector, above which we are outside the set.
        Ex: (3, 5, 2) means that the set is made up of the cube [0, 3] x [0, 5] x [0, 2]
        default: None
    """
    def __init__(self, states: set=None, state_boundaries: tuple or int or float=None):
        if states is None and state_boundaries is None:
            raise ValueError("Either parameter `states` must be not None or `state_boundaries` must be not None")
        #-- Parameter `states`
        # Check the dimension of each state given in the set: they should all have the same dimension
        if states is not None:
            if not isinstance(states, set):
                raise ValueError("Input parameter `states` must be a set: {}".format(states))
            if len(states) > 0:
                s0 = list(states)[0]
                is_problem_1d = is_scalar(s0)
                for s in states:
                    if  is_problem_1d and not is_scalar(s) or \
                        not is_problem_1d and len(s) != len(s0):
                        if is_problem_1d:
                            dim0 = 1
                        else:
                            dim0 = len(s0)
                        raise ValueError("Not all the elements in the set of states have the same dimension (dim. of one element = {}, dim. just found = {})" \
                                         .format(dim0, len(s)))
        self.states = states

        #-- Parameter `state_boundaries`
        if is_scalar(state_boundaries):
            # Convert the scalar to a tuple in order to facilitate handling unidimensional and multidimensional states
            # (the class assumes that `state_boundaries` is a tuple throughout its process)
            # Note that when the parameter is passed as e.g. `(3)` (note there is no comma as there is in `(3,)`)
            # it is converted to a scalar, i.e. it is NOT interpreted as a tuple
            state_boundaries = (state_boundaries,)
        if state_boundaries is not None and not all([is_integer(s) for s in state_boundaries]):
            raise ValueError("The values in parameter `state_boundaries` values must be integers "
                             "(because they represent boundaries of integer-valued state dimensions: {}" \
                             .format(state_boundaries))
        self.state_boundaries = state_boundaries

    def random_choice(self, size: int=None):
        """
        Chooses one or more random states from the set

        The result is different depending on whether the storage format of the set (see method `getStorageFormat()`)
        is "states" (where the states in the set are explicitly listed) or "state_boundaries" (where the states in the
        set are defined implicitly by the integer upper limits in each dimension of a state), in which case all states
        made up of integer values ONE of which satisfies one of those upper limits belong to the set.
        Example of the latter: if the state_boundaries are [4, 2, 5], then following are examples of states belonging
        to the set: [4, 0, 0], [3, 2, 1], [3, 2, 5], etc.

        Arguments:
        size: int or None
            The size of the sample of states.
            default: None

        Return: scalar or list of scalar or tuples
        If `size` is None, a scalar or tuple with the selected single element of the set.
        Otherwise, list with as many scalar or tuple elements of the set as specified by `size`,
        defining the randomly chosen states among all possible states in the set.
        """
        size_orig = size
        if size_orig is None:
            size = 1

        if self.states is not None:
            # Note the following link about randomly choosing elements from a set:
            # https://stackoverflow.com/questions/1262955/how-do-i-pick-2-random-items-from-a-python-set
            idxs_chosen = np.random.choice(len(self.states), size=size)
            chosen_states = [list(self.states)[idx] for idx in idxs_chosen]
        else:
            # First we randomly choose in which dimension we are at the state_value
            idx_state_boundaries = np.random.choice(len(self.state_boundaries), size=size)
            # Now we freely choose the other dimensions between 0 and the state_value once one of the state_boundaries has been chosen
            chosen_states = [[]]*size
            for s in range(size):
                chosen_states[s] = [-1]*len(self.state_boundaries)
                chosen_states[s][ idx_state_boundaries[s] ] = self.state_boundaries[ idx_state_boundaries[s] ]
                other_dimensions_to_choose_a_state_value = [ss for ss in range(len(self.state_boundaries)) if ss != idx_state_boundaries[s]]
                for idx in other_dimensions_to_choose_a_state_value:
                    # Random number between 0 and state_boundaries[idx]
                    chosen_states[s][idx] = np.random.choice(self.state_boundaries[idx] + 1)
                if len(chosen_states[s]) == 1:
                    # Convert the chosen state to scalar
                    chosen_states[s] = chosen_states[s][0]
                else:
                    # Convert each list to tuple because the selected states are not mutable (lists are mutable, tuples are not)
                    chosen_states[s] = tuple(chosen_states[s])

        if size_orig is None:
            return chosen_states[0]
        else:
            return chosen_states

    def getStates(self):
        if self.getStorageFormat() == "states":
            if len(self.states) == 1:
                return self.states[0]
            else:
                return self.states
        else:
            if len(self.state_boundaries) == 1:
                return self.state_boundaries[0]
            else:
                return self.state_boundaries

    def getStorageFormat(self):
        "Returns the format in which the states in the set are stored, either in their `states` attribute or in their `state_boundaries` attribute"
        if self.states is not None:
            return "states"
        else:
            return "state_boundaries"

    def getStateDimension(self):
        if self.getStorageFormat() == "states":
            if len(self.states) == 0:
                return None
            else:
                return 1 if is_scalar(list(self.states)[0]) else len(list(self.states)[0])
        else:
            if len(self.state_boundaries) == 0:
                return None
            else:
                return 1 if is_scalar(self.state_boundaries) else len(self.state_boundaries)

    def getNumStates(self):
        """
        Get the number of states in the set when the storage format is 'states' or an upper bound of the number of states
        when the storage format is 'state_boundaries' and the state dimension is larger than 1.

        Computing the exact number of states requires further thinking which I have not the time to do now (09-Jan-2023).

        This method however is mostly used to check whether the set of states is just 1, for assertion purposes.
        """
        if self.getStorageFormat() == "states":
            return len(self.states)
        else:
            if len(self.state_boundaries) == 0:
                return 0
            else:
                if self.getStateDimension() == 1:
                    return 1
                else:
                    # Compute an upper bound of the number of states in the absorption set
                    # This is an upper bound because we are not subtracting the repeated states in this count, such as:
                    # if state_boundaries = [3, 2, 4], then we are counting more than once all the states having more than
                    # one dimension at the boundary (e.g. [3, 2, 1] or [0, 2, 4], etc.)
                    n_states_upper_bound = 0
                    for i in range(self.getStateDimension()):
                        n_states_upper_bound += np.prod([b+1 for ii, b in enumerate(self.state_boundaries) if ii != i])
                    return n_states_upper_bound

    # toString() method: i.e. what should be shown when including an object of this class in a print() call
    def __str__(self):
        return self.getStates()


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
            - 'theta': the initial theta parameter value for the parameterized policy.
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
    policies = dict({PolicyTypes.ACCEPT: dict_params_policy['parameterized_policy'](env_queue, theta=dict_params_policy['theta'])
                                                if is_scalar(dict_params_policy['theta'])
                                                else [policy(env_queue, theta=dict_params_policy['theta'][i]) for i, policy in enumerate(dict_params_policy['parameterized_policy'])],
                     PolicyTypes.ASSIGN: policy_assign})

    # Learners (for V, Q, and P)
    learnerV = dict_params_learners['V']['learner'](env_queue,
                                                    gamma=dict_params_learners['V']['params'].get('gamma'))
    learnerQ = None
    if dict_params_learners['P']['learner'] is None:
        learnerP = None
    else:
        learnerP = dict_params_learners['P']['learner'](env_queue,
                                                        policies[PolicyTypes.ACCEPT],
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
        Discrete time step of the step carried out now.

    env: environment
        The environment where the agent acts.
        NOTE: This object is updated after the step.

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

def analyze_event_times(rates: Union[list, tuple, np.ndarray], times, groups, group_name="group", plot=False):
    """
    Computes and plots the event times (e.g. inter-arrival times or service times) for each of the given groups,
    and compares their rates with the nominal rates of each group.

    Ex 1: If a system has two arriving job classes and three servers, parameter `rates` could be:
    - the inter-arrival rates for each of two job classes (groups) in ONE PARTICULAR server.
    - the service rates of each of three servers, which act as groups.
    Parameters `times` and `groups` should have the same lengths and they could contain:
    - the inter-arrival times for each of the two job classes, where the class to which the observed inter-arrival time
    belongs to is indicated by the `groups` list. Ex: groups = [0, 0, 1, 0, 1]; times = [0.3, 0.2, 0.6, 0.1, 0.15]
    which means that the inter-arrival times observed for class '0' are [0.3, 0.2, 0.1] and the inter-arrival times
    observed for class '1' are [0.6, 0.15].
    - the service rates for each of the three servers, following the same logic just described for the inter-arrival times.

    Ex 2: If a system has one job class and one server, parameter `rates` could be the rates for each of N replications
    of the system simulation (e.g. N particles) and parameters `times` and `groups` could be:
    - the inter-arrival times observed in each of the N particles and the particle numbers associated to them.
    - the service times observed in each of the N particles and the particle numbers associated to them.
    e.g. times = [0.3, 0.2, 0.6, 0.1, 0.15], groups = [0, 0, 0, 1, 1]
    meaning that the first three times were observed in particle 0 and the last two times were observed in particle 1.

    Arguments:
    rates: list
        List containing the *true* jump rates of the Markov process whose observed times are analyzed.
        Its length is NOT related with the length of parameters `times` and `groups`.
        See examples above.

    times: list
        List containing the times to analyze by comparing them with the expected times deduced from the given `rates`,
        in terms of their average and standard deviation / standard error.
        It should have the same length as the `groups` list.
        See examples above.

    groups: list
        It should have the same length as the `groups` list and its value be integer values that index the `rates` list.
        See examples above.

    group_name: (opt) str
        Name that is used to give a meaning to the `groups` list values.
        Ex:
        - "job class" if `times` contains inter-arrival times for different job classes.
        - "particle" if `times` contains inter-arrival times or service times observed in different particles
        of a Fleming-Viot system.
        default: "group" (i.e. a name that contains no semantic)

    plot: (opt) bool
        Whether to generate a histogram of the values given in `times` for each group given in `rates`.
        default: False
    """
    print("Distribution of event times by {}:".format(group_name))
    if plot:
        fig, axes = plt.subplots(1, len(rates), squeeze=False)
    for j, r in enumerate(rates):
        _times = [t for i, t in zip(groups, times) if i == j]
        print(group_name + " {}: true mean (std) = {:.3f} ({:.3f}), observed mean (std, SE) (n={}) = {:.3f} ({:.3f}, {:.3f})" \
              .format(j, 1 / r, 1 / r, len(_times), np.mean(_times), np.std(_times), np.std(_times)/np.sqrt(len(_times))))
        if plot:
            axes[0][j].hist(_times)
            axes[0][j].set_title("Dist. event times: " + group_name + " {}, n = {}, mean = {:.3f}, std = {:.3f}, SE = {:.3f}" \
                                 .format(j, len(_times), np.mean(_times), np.std(_times), np.std(_times)/np.sqrt(len(_times))))
