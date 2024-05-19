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
from Python.lib.agents.policies import PolicyTypes

from Python.lib.environments.queues import GenericEnvQueueWithJobClasses
from Python.lib.estimators.fv import SurvivalProbabilityEstimation

from Python.lib.utils.basic import is_scalar, is_integer
from Python.lib.utils.computing import all_combos_with_max_limits


@unique
class LearningMethod(Enum):
    MC = 1
    FV = 2

# Default number of burn-in time steps to exclude from the estimation of expectations
# (e.g. E(T) in Monte-Carlo estimation and E(T_A) in Fleming-Viot estimation) in order to assume the process is stationary
BURNIN_TIME_STEPS = 20
# Default minimum number of observed cycles (under assumed stationarity) that should be used to estimate expectations under
# stationarity (e.g. stationary probabilities, E(T) in Monte-Carlo, E(T_A) in Fleming-Viot)
MIN_NUM_CYCLES_FOR_EXPECTATIONS = 5

DEBUG_TRAJECTORIES = False


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

    set_boundaries: (opt) tuple of non-negative int, non-negative int or non-negative float
        A list defining the state value in each dimension of the state vector, above which we are outside the set.
        Ex: (3, 5, 2) means that the set is made up of the cube [0, 3] x [0, 5] x [0, 2]
        default: None
    """
    def __init__(self, states: set=None, set_boundaries: Union[tuple, int, float]=None):
        if states is None and set_boundaries is None:
            raise ValueError("Either parameter `states` must be not None or `set_boundaries` must be not None")
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

        #-- Parameter `set_boundaries`
        if is_scalar(set_boundaries):
            # Convert the scalar to a tuple in order to facilitate handling unidimensional and multidimensional states
            # (the class assumes that `set_boundaries` is a tuple throughout its process)
            # Note that when the parameter is passed as e.g. `(3)` (note there is no comma as there is in `(3,)`)
            # it is converted to a scalar, i.e. it is NOT interpreted as a tuple
            set_boundaries = (set_boundaries,)
        if set_boundaries is not None and not all([is_integer(s) and s >= 0 for s in set_boundaries]):
            raise ValueError("The values in parameter `set_boundaries` must be non-negative integers "
                             "(because they represent the upper boundaries relative to 0 of integer-valued state dimensions: {}" \
                             .format(set_boundaries))
        self.set_boundaries = set_boundaries

    def random_choice(self, size: int=None):
        """
        Chooses one or more random states from the set

        The result is different depending on whether the storage format of the set (see method `getStorageFormat()`)
        is "states" (where the states in the set are explicitly given --see also method `states_are_given_explicitly()`)
        or "set_boundaries" (where the states in the set are defined implicitly by the integer upper limits in each
        dimension of a state), in which case all states made up of integer values, of which at least ONE is equal to
        one of those upper limits, belong to the set.
        Example of the latter: if the set_boundaries are [4, 2, 5], then the following are examples of states belonging
        to the set:
        [4, 0, 0] (the first element is at the boundary)
        [3, 2, 1] (the second element is at the boundary)
        [3, 2, 5] (both the second and third elements are at the boundary)

        Arguments:
        size: int or None
            The size of the sample of states.
            default: None

        Return: scalar or list of scalar or tuples
        If `size` is None, a scalar or tuple with the selected single element of the set.
        Otherwise, list with as many scalar or tuple elements of the set as specified by `size`,
        defining the randomly chosen states among all possible states in the set.
        """
        at_least_one_dimension_is_at_boundary = lambda state: np.sum(np.array(state) == np.array(self.set_boundaries)) >= 1

        size_orig = size
        if size_orig is None:
            size = 1

        if self.states_are_given_explicitly():
            # Note the following link about randomly choosing elements from a set:
            # https://stackoverflow.com/questions/1262955/how-do-i-pick-2-random-items-from-a-python-set
            idxs_chosen = np.random.choice(len(self.states), size=size)
            chosen_states = [list(self.states)[idx] for idx in idxs_chosen]
        else:
            # States are given indirectly via the set boundaries (e.g. [4, 2, 5])
            # => First we randomly choose which dimension to set at the boundary (e.g. if dimension 0 is chosen, then we set state[0] = 4)
            idx_set_boundaries = np.random.choice(len(self.set_boundaries), size=size)
            # Now we freely choose the other dimensions between 0 and the corresponding state value (e.g. we choose dimension 1 between 0 and 2, and dimension 2 between 0 and 5)
            chosen_states = [[]]*size
            for s in range(size):
                chosen_states[s] = [-1]*len(self.set_boundaries)
                chosen_states[s][ idx_set_boundaries[s] ] = self.set_boundaries[ idx_set_boundaries[s] ]
                other_dimensions_to_choose_a_state_value = [ss for ss in range(len(self.set_boundaries)) if ss != idx_set_boundaries[s]]
                for idx in other_dimensions_to_choose_a_state_value:
                    # Random number between 0 and set_boundaries[idx]
                    chosen_states[s][idx] = np.random.choice(self.set_boundaries[idx] + 1)
                if len(chosen_states[s]) == 1:
                    # Convert the chosen state to scalar
                    chosen_states[s] = chosen_states[s][0]
                else:
                    # Convert each list to tuple because the selected states are not mutable (lists are mutable, tuples are not)
                    chosen_states[s] = tuple(chosen_states[s])
            assert all([at_least_one_dimension_is_at_boundary(state) for state in chosen_states]), \
                "All chosen states have at least one dimension at its boundary"

        if size_orig is None:
            return chosen_states[0]
        else:
            return chosen_states

    def getStates(self, exactly_one_dimension_at_boundary=False, at_least_one_dimension_at_boundary=False):
        """
        Returns the set of all possible states, explicitly enumerated, regardless of the storage format used in the object

        Parameters `exactly_one_dimension_at_boundary` and `at_least_one_dimension_at_boundary` can be used to restrict
        the set of states returned to those satisfying the condition indicated by the parameter.

        They cannot be both True and the same time.
        """
        if self.getStorageFormat() == "states":
            if len(self.states) == 1:
                return self.states[0]
            else:
                return self.states
        else:
            # We explicitly enumerate all states belonging to the set
            if exactly_one_dimension_at_boundary and at_least_one_dimension_at_boundary:
                raise ValueError("Parameters exactly_one_dimension_at_boundary and at_least_one_dimension_at_boundary cannot be both True")
            states = set()
            combos_generator = all_combos_with_max_limits(self.set_boundaries)
            while True:
                try:
                    next_combo = next(combos_generator)
                    if exactly_one_dimension_at_boundary or at_least_one_dimension_at_boundary:
                        match_combo_to_boundaries = np.array(next_combo) == np.array(self.set_boundaries)
                    if  not exactly_one_dimension_at_boundary and not at_least_one_dimension_at_boundary or \
                        exactly_one_dimension_at_boundary and np.sum(match_combo_to_boundaries) == 1 or \
                        at_least_one_dimension_at_boundary and np.sum(match_combo_to_boundaries) >= 1:
                        states.add( next_combo[0] if len(next_combo) == 1 else tuple(next_combo) )
                        ## NOTE: Since next_combo is a list and the value added to the `states` set is a tuple
                        ## they won't share the same memory address and there is no risk that all elements will be the same in the end.
                except StopIteration:
                    break
            combos_generator.close()
            return states

    def getSetBoundaries(self):
        if self.getStorageFormat() == "states":
            raise AttributeError("States in the set are NOT stored implicitly by defining the set boundaries in each dimension")
        else:
            if len(self.set_boundaries) == 1:
                return self.set_boundaries[0]
            else:
                return self.set_boundaries

    def getStorageFormat(self):
        "Returns the format in which the states in the set are stored, either in their `states` attribute or in their `set_boundaries` attribute"
        if self.states is not None:
            return "states"
        else:
            return "set_boundaries"

    def getStateDimension(self):
        if self.getStorageFormat() == "states":
            if len(self.states) == 0:
                return None
            else:
                return 1 if is_scalar(list(self.states)[0]) else len(list(self.states)[0])
        else:
            if len(self.set_boundaries) == 0:
                return None
            else:
                return 1 if is_scalar(self.set_boundaries) else len(self.set_boundaries)

    def getNumStates(self):
        """
        Get the number of states in the set when the storage format is 'states' or an upper bound of the number of states
        when the storage format is 'set_boundaries' and the state dimension is larger than 1.

        Computing the exact number of states requires further thinking which I have not the time to do now (09-Jan-2023).

        This method however is mostly used to check whether the set of states is just 1, for assertion purposes.
        """
        if self.getStorageFormat() == "states":
            return len(self.states)
        else:
            if len(self.set_boundaries) == 0:
                return 0
            else:
                if self.getStateDimension() == 1:
                    return 1
                else:
                    # Compute an upper bound of the number of states in the absorption set
                    # This is an upper bound because we are not subtracting the repeated states in this count, such as:
                    # if set_boundaries = [3, 2, 4], then we are counting more than once all the states having more than
                    # one dimension at the boundary (e.g. [3, 2, 1] or [0, 2, 4], etc.)
                    n_states_upper_bound = 0
                    for i in range(self.getStateDimension()):
                        n_states_upper_bound += np.prod([b+1 for ii, b in enumerate(self.set_boundaries) if ii != i])
                    return n_states_upper_bound

    def states_are_given_explicitly(self):
        return self.getStorageFormat() == "states"

    # toString() method: i.e. what should be shown when including an object of this class in a print() call
    def __str__(self):
        return self.getStates()


#-------------------------------------------- FUNCTIONS --------------------------------------#
def choose_state_from_setofstates_based_on_distribution(set_of_states: SetOfStates, dist_proba: dict,
                                                        exactly_one_dimension_at_boundary=False,
                                                        at_least_one_dimension_at_boundary=False):
    "Chooses a state from the set of states using the given probability distribution, truncated to those states"
    if exactly_one_dimension_at_boundary and at_least_one_dimension_at_boundary:
        raise ValueError("Parameters exactly_one_dimension_at_boundary and at_least_one_dimension_at_boundary cannot be both True")

    all_states = set_of_states.getStates(   exactly_one_dimension_at_boundary=exactly_one_dimension_at_boundary,
                                            at_least_one_dimension_at_boundary=at_least_one_dimension_at_boundary)
    if not all_states.issubset(set(dist_proba.keys())):
        raise ValueError("All states to select from must be present in the dictionary containing their probability distribution: {}".format(all_states))

    # Truncate the given distribution to the given set of states
    states, dist_states = zip(*filter(lambda state_proba_pair: state_proba_pair[0] in all_states, dist_proba.items()))
    dist = dist_states / np.sum(dist_states)
    assert np.isclose(np.sum(dist), 1.0), \
        "The probabilities in the distribution on which start states in the activation set are sampled sums up to 1: {}" \
        .format(np.sum(dist))

    # Choose a state following the given distribution
    idx_selected_value = np.random.choice(len(states), p=dist)
    chosen_state = states[idx_selected_value]

    if exactly_one_dimension_at_boundary:
        assert sum(np.array(chosen_state) == np.array(set_of_states.getSetBoundaries())) == 1, \
            "Exactly ONE dimension of the start state must be at its limit value (boundary) defined by {}: {}" \
                .format(set_of_states.getSetBoundaries(), chosen_state)
    elif at_least_one_dimension_at_boundary:
        assert np.sum(np.array(chosen_state) == np.array(set_of_states.getSetBoundaries())) >= 1, \
            "At least one dimension must be at its limit (boundary) defined by the absorption set ({}): {}" \
            .format(set_of_states.getSetBoundaries(), chosen_state)

    return chosen_state


def is_state_in_setofstates(state, set_of_interest: SetOfStates):
    "Check whether the QUEUE state of the system is in the set of queue states of interest"
    assert set_of_interest is not None
    if set_of_interest.getStorageFormat() == "states":
        states_of_interest = set_of_interest.getStates()
        return state in states_of_interest
    else:
        # The set is assumed to be defined by its boundaries in each dimension and include all states whose dimension value is smaller than or equal to the respective boundary
        set_of_interest_boundaries = set_of_interest.getSetBoundaries()
        # Note: we convert lists to arrays in order to perform an element-wise comparison of their elements
        particle_state = np.array([state]) if is_scalar(state) else np.array(state)
        set_boundaries = np.array([set_of_interest_boundaries]) if is_scalar(set_of_interest_boundaries) else np.array(set_of_interest_boundaries)
        if  any(particle_state == set_boundaries) and \
            all(particle_state <= set_boundaries):
            return True
        else:
            return False


def choose_state_from_set(_set, dist_proba: dict=None):
    """
    Chooses a state from a set, optionally following a given distribution

    Arguments:
    _set: set
        Set from where a state must be chosen.

    dist_proba: dict
        Dictionary indexed by the states in `_set`.
        When given, it must have the same number of elements as `_set`.
        default: None

    Return: state in _set
    The state chosen from the set of states `_set`, which is chosen uniformly at random when dist_proba=None or
    following the distribution in the `dist_proba` dictionary otherwise.
    """
    if not isinstance(_set, set):
        raise ValueError("Parameter `_set_` must be a set ({})".format(type(_set)))

    if dist_proba is None:
        # Uniformly selected state
        start_state = np.random.choice(sorted(_set)) if len(_set) > 1 else list(_set)[0]
    else:
        # State is selected following the given distribution
        # Note that we sort the dictionary keys because `list(_set)` returns the elements in `_set` sorted.
        if not isinstance(dist_proba, dict):
            raise ValueError("The probability distribution of states must be a dictionary ({})".format(type(dist_proba)))
        start_state = np.random.choice(sorted(_set), p=[dist_proba[k] for k in sorted(dist_proba.keys())])
    return start_state


def check_done(tmax, t, state, action, reward):
    """
    Checks whether the simulation is done

    tmax: int
        Maximum discrete time allowed for the simulation.

    t: int
        Current simulation time.

    state: Environment dependent
        S(t): state of the environment at time t, BEFORE the action is taken.

    action: Environment dependent
        A(t): action received by the environment at time t.

    reward: float
        R(t+1): reward yielded by the environment after taking action A(t) at state S(t).

    Return: bool
        Whether the simulation is done because the maximum number of iterations has been reached.
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
    - observation: the next state of the environment after the action is taken
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
    return verbose and np.mod(t_learn, verbose_period) == 0

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


def parse_simulation_parameters(dict_params_simul, env):
    """
    Parses input parameters received by the estimate_blocking_*() functions

    Arguments:
    dict_params_simul: dict
        Dictionary of simulation and estimation parameters to parse.
        It should contain at least the following keys:
        - T: either the number of simulation steps to use in the Monte-Carlo simulation when estimating probabilities
        using Monte-Carlo or the number of arrival events to use in the Monte-Carlo simulation used in
        the FV estimator of probabilities to estimate E(T_A), i.e. the expected reabsorption time of a single Markov chain,

        The following keys are optional:
        - If given, one of the following set of keys:
            a) for queue systems with a single buffer governing blocking (e.g. single-server, multi-server with single buffer)
            holding jobs that await being served. If we call this value J, then J-1 is the absorption buffer size.
                - buffer_size_activation: the overall system's "buffer size for activation" for systems with a single buffer.
            b) for all other systems (e.g. network systems where blocking occurs at multi-dimensional states, such as a loss network with multi-class jobs):
                - absorption_set: SetOfStates object defining the set of absorption states.
                - activation_set: SetOfStates object defining the set of activation states.
            When both (a) and (b) are given in the dictionary, and the system is a queue system, precedence is given to
            the information provided in (a).
        - burnin_time_steps: number of burn-in time steps to allow at the beginning of the simulation in order to
        consider that the Markov process is in stationary regime.
        default: BURNIN_TIME_STEPS
        - min_num_cycles_for_expectations: minimum number of observed reabsorption cycles to consider that the expected
        reabsorption cycle time, E(T_A), can be reliably estimated.
        default: MIN_NUM_CYCLES_FOR_EXPECTATIONS
        - method_survival_probability_estimation: method to use to estimate the survival probability P(T>t)
        in the Fleming-Viot estimation approach, which can be either:
            - SurvivalProbabilityEstimation.FROM_N_PARTICLES, where the survival probability is estimated from the first
            absorption times of each of the N particles used in the FV simulation, or
            - SurvivalProbabilityEstimation.FROM_M_CYCLES, where the survival probability is estimated from the M cycles
            defined by the return to the absorption set A of a single particle being simulated. The number M is a
            function of the number of arrival events T and of J, the size of the absorption set A. For a fixed
            T value, M decreases as J increases as the queue has less probability of visiting a state further away from 0.

            ***FROM_N_PARTICLES is the the preferred method*** because of the non-control of the number of cycles M in
            the other case, whereas N is controlled by the parameter defining the number of particles used in the FV simulation.
            If we increase N, the simulation time for FV will be larger as this is determined by the maximum observed survival time
            (as no contribution to the integral appearing in the FV estimator (int{P(T>t)*phi(t)}) is received from
            events happening past the maximum observed survival time).
        default: SurvivalProbabilityEstimation.FROM_N_PARTICLES
        - seed: the seed to use for the pseudo-random number generation.
        default: None

    env: Environment object
        Environment that should be used to compute the burn-in time (i.e. absolute time) from parameter 'burnin_time_steps',
        which is done using the compute_burnin_time_from_burnin_time_steps() function that uses the expected inter-event time
        computed from the job arrival rates and service rates of the queue.

    Return: dict
    Input dictionary with the following parsed parameters that are normally used in the simulation process, depending on the environment:
    - T                                         (all environments)
    - absorption_set                            (all environments)
    - activation_set                            (all environments)
    - burnin_time_steps                         (all environments)
    - burnin_time                               (queue environment)
    - min_num_cycles_for_expectations           (all environments)
    - method_survival_probability_estimation    (all environments)
    - seed                                      (all environments)
    """
    set_required_simul_params = set({'T'})

    if isinstance(env, GenericEnvQueueWithJobClasses):
        if 'buffer_size_activation' in dict_params_simul.keys():
            if dict_params_simul['buffer_size_activation'] < 1:
                raise ValueError("The activation buffer size must be at least 1: {}".format(dict_params_simul['buffer_size_activation']))
            # Define the simulation parameters that will be used from now on below in a unified estimation process that
            # encompasses both the case where blocking occurs at a single buffer size (e.g. single-server system or
            # multi-server system with single buffer) or when it occurs at a set of states (e.g. loss network with multi-class jobs).
            dict_params_simul['absorption_set'] = SetOfStates(set_boundaries=dict_params_simul['buffer_size_activation'] - 1)
            dict_params_simul['activation_set'] = SetOfStates(set_boundaries=dict_params_simul['buffer_size_activation'])
    if not set_required_simul_params.issubset(dict_params_simul.keys()):
        raise ValueError("Not all required parameters were given in `dict_params_simul`, which requires: {}\nGiven: {}".format(set_required_simul_params, dict_params_simul.keys()))

    # Parse the remaining (optional) simulation parameters
    dict_params_simul['burnin_time_steps'] = dict_params_simul.get('burnin_time_steps', BURNIN_TIME_STEPS)
    # Continuous burn-in time used to filter the continuous-time cycle times observed during the single-particle simulation
    # that is used to estimate the expected reabsorption cycle time, E(T_A).
    if isinstance(env, GenericEnvQueueWithJobClasses):
        # TODO: (2023/09/06) Try to remove the circular import that this import implies (because simulators.queues imports functions or constants from this file (e.g. BURNIN_TIME_STEPS)
        from Python.lib.simulators.queues import compute_burnin_time_from_burnin_time_steps
        dict_params_simul['burnin_time'] = compute_burnin_time_from_burnin_time_steps(env, dict_params_simul['burnin_time_steps'])
        print("Computed burn-in time from parameter burnin_time_steps = {}: {}".format(dict_params_simul['burnin_time_steps'], dict_params_simul['burnin_time']))

    dict_params_simul['min_num_cycles_for_expectations'] = dict_params_simul.get('min_num_cycles_for_expectations', MIN_NUM_CYCLES_FOR_EXPECTATIONS)
    dict_params_simul['method_survival_probability_estimation'] = dict_params_simul.get('method_survival_probability_estimation', SurvivalProbabilityEstimation.FROM_N_PARTICLES)
    dict_params_simul['seed'] = dict_params_simul.get('seed')

    return dict_params_simul
#-------------------------------------------- FUNCTIONS --------------------------------------#
