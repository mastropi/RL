# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 22:15:48 2020

@author: Daniel Mastropietro
@description: Functions used to perform specialized computations commonly used by the application.
"""

import warnings
import copy
from unittest import TestCase
from typing import Union

from collections import deque
import numpy as np
import pandas as pd

from Python.lib.utils.basic import as_array, is_integer


def mad(x):
    """
    Computes the Median Absolute Deviation on a list or array of values.
    
    Missing values are ignored.

    x: list or numpy array
        Values on which the MAD is computed.

    Return: float
        median( |x(i) - median(x)| )
    """
    
    # Convert to an array if x is given as list
    x = np.array(x)
    x_median = np.nanmedian(x)
    x_dev = x - x_median
    x_mad = np.median( np.abs( x_dev ) )
    
    return x_mad


# Function that returns a function to compute percentiles using the agg() aggregation method in pandas `groupby` data frames
# Ref: https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
def percentile(n):
    def percentile_(x):
        return x.quantile(n / 100)

    percentile_.__name__ = 'percentile_{:.0f}'.format(n)
    return percentile_


def rmse(Vtrue: np.ndarray, Vest: np.ndarray, weights: np.ndarray=None):
    """Root Mean Square Error (RMSE) between Vtrue and Vest, optionally weighted

    All weights are assumed to be non-negative.

    Arguments:
    Vtrue: np.ndarray of any shape (the same shape as Vest and weights)
        True Value function.

    Vest: np.ndarray of any shape (the same shape as Vtrue and weights)
        Estimated Value function.

    weights: np.ndarray of any shape (the same shape as Vtrue and Vest)
        Weights to use in the computation of the MAPE.
        Ex: Number of visits associated to each estimated value.

    Return: float
    For the weighted version:
        sqrt( sum( weight * (Vest - Vtrue)**2 ) / sum(weight) )
    For the unweighted version:
        sqrt( mean( (Vest - Vtrue)**2 ) )
    """
    if type(Vtrue) != np.ndarray or type(Vest) != np.ndarray or (weights is not None and type(weights) != np.ndarray):
        raise ValueError("The first three input parameters must be numpy arrays (`weights` can be None)")
    if Vtrue.shape != Vest.shape or (weights is not None and Vest.shape != weights.shape):
        raise ValueError("The first three input parameters have the same shape({}, {}, {})" \
                         .format(Vtrue.shape, Vest.shape, weights and weights.shape or ""))

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    if weights is not None:
        mse = np.sum( weights * (Vest - Vtrue)**2 ) / np.sum(weights)
    else:
        mse = np.mean( (Vest - Vtrue)**2 )

    return np.sqrt(mse)


def mape(Vtrue: np.ndarray, Vest: np.ndarray, weights: np.ndarray=None):
    """Mean Absolute Percent Error (MAPE) between Vtrue and Vest, weighted or not weighted by weights.

    All weights are assumed to be non-negative.

    Arguments:
    Vtrue: np.ndarray of any shape (the same shape as Vest and weights)
        True Value function.

    Vest: np.ndarray of any shape (the same shape as Vtrue and weights)
        Estimated Value function.

    weights: np.ndarray of any shape (the same shape as Vtrue and Vest)
        Weights to use in the computation of the MAPE.
        Ex: Number of visits associated to each estimated value.

    Return: float
    The error in each element is set to 0.0 when the two compared values Vtrue and Vest are equal to 0.0
    (so that we avoid a NaN when doing (0.0 - 0.0) / 0.0).

    So, excluding these values, the returned value is computed as follows:
    - for the weighted version:
        sum( weight * abs(Vest - Vtrue) / abs(Vtrue) ) / sum(weight)
    - for the unweighted version:
        mean( abs(Vest - Vtrue) / abs(Vtrue) )
    If any value in Vtrue is 0.0 and the corresponding estimated value Vest is not 0.0,
    NaN is returned for the MAPE.
    """
    if type(Vtrue) != np.ndarray or type(Vest) != np.ndarray or (weights is not None and type(weights) != np.ndarray):
        raise ValueError("The first three input parameters must be numpy arrays (`weights` can be None)")
    if Vtrue.shape != Vest.shape or (weights is not None and Vest.shape != weights.shape):
        raise ValueError("The first three input parameters have the same shape({}, {}, {})" \
                        .format(Vtrue.shape, Vest.shape, weights and weights.shape or ""))

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    # Convert all arrays to a 1D array to avoid problems when zip()-ing them in the list below
    nobs = np.prod(Vtrue.shape)
    Vtrue_1d = np.squeeze(Vtrue.reshape(1, nobs))
    Vest_1d = np.squeeze(Vest.reshape(1, nobs))
    if weights is not None:
        weights_1d = np.squeeze(weights.reshape(1, nobs))

    if weights is not None:
        weighted_error_rel = [0.0   if x_true == x_est == 0.0 or w == 0.0
                                    else w * np.abs(x_est - x_true) / np.abs(x_true)
                                    for x_true, x_est, w in zip(Vtrue_1d, Vest_1d, weights_1d)]
        MAPE = np.sum( weighted_error_rel ) / np.sum(weights[:])
    else:
        error_rel = [0.0    if x_true == x_est == 0.0
                            else np.Inf if x_true == 0.0
                                        else np.abs(x_true - x_est) / np.abs(x_true)
                            for x_true, x_est in zip(Vtrue_1d, Vest_1d)]
        MAPE = np.mean(error_rel)

    return MAPE


def smooth(x, window_size):
    "Smooth a signal by moving average"
    # Ref: https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
    if window_size <= 1:
        return x
    else:
        # Convert to a pandas series
        x_series = pd.Series(x)
        # Get the window of series of observations till the current time
        windows = x_series.rolling(window_size)
        # Smooth!
        x_smooth = windows.mean()

        return x_smooth


def comb(n, k):
    """
    Efficient calculation of the combinatorial number, which does NOT use the factorial.
    
    Ex: `comb(C+K-1,K-1)` does NOT give overflow but `factorial(C+K-1) / factorial(K-1) / factorial(C)`
    does for e.g. C = 400, K = 70
    The actual error is "OverflowError: integer division result too large for a float".
    """
    if  not is_integer(n) and not is_integer(k):
        raise ValueError("n and k must be integer (n={}, k={})".format(n,k))
    if (n < k):
        raise ValueError("n cannot be smaller than k (n={}, k={})".format(n,k))

    num = 1
    den = 1
    for i in range( min(n-k, k) ):
        num *= n - i
        den *= i + 1

    return int( num / den )


def factorial(n):
    "Compute the factorial of a number"
    if not is_integer(n) or n < 0:
        raise ValueError("The input parameter `n` must be a non-negative integer ({}, type={})".format(n, type(n)))
    res = 1
    for i in range(1, n+1):
        res *= i
    return res


def all_combos_with_sum(R, C):
    """
    Returns a generator of all possible integer-valued lists whose elements sum up to a fixed non-negative integer

    Arguments:
    R: int
        Dimension of the integer-valued lists to generate. E.g. if R = 3, a generated array will be of the form [x, y, z].

    C: int
        Sum of the elements of the integer-valued list.
    """

    # Note: documentation on the `yield` expression: https://docs.python.org/3/reference/expressions.html#yieldexpr

    def all_combos_with_sum_in_sublist(v, dim):
        # print("\ndim={}:".format(dim))
        r = dim
        # Sum over the indices of v to the left of r
        # which determine the new capacity to satisfy on the indices from r to the right
        vleft = sum(v[0:r])
        if r < R - 1:
            # Set the current dimension value to all its possible values
            # and for each of those values solve the sub-problem of finding all valid combinations
            # for the sub-list that is to the RIGHT of r (this step calls this same function recursively)
            for k in range(C - vleft, -1, -1):
                # print("\tFOR k ({} downto 0): dim: {}, k={}".format(C-vleft, r, k))
                # print("\tv before={}".format(v))
                v[r] = k
                # print("\tv={}".format(v))
                # print("new call: (offset={})".format(r+1))
                # NOTE THE USE OF `yield from` in order to get an yield value from a recursive call!
                yield from all_combos_with_sum_in_sublist(v, r + 1)
        else:
            # No degrees of freedom left for the last dimension (right-most index)
            # print("\tv before={}".format(v))
            v[r] = C - vleft
            # print("\tv={}".format(v))
            # print("yield v! {}".format(v))
            assert sum(v) == C, "The elements of v sum up to C={} (v={})".format(C, v)
            yield v

    if C < 0 or not is_integer(C):
        raise ValueError("Parameter `C` specifying the sum to be satisfied by each valid combinations should be a non-negative integer: {}".format(C))

    # Initialize the list on the combination that is valid for sure (i.e. list with all zeros, as the sum C must be at least 0)
    v = [0] * R
    # We start the combination generation by calling "the generator that solves every sub-problem" on the WHOLE list (i.e. on the list with its full dimension)
    gen = all_combos_with_sum_in_sublist(v, 0)
    return gen


def all_combos_with_max_limits(L: Union[list, tuple]):
    """
    Returns a generator of all possible integer-valued lists whose elements are limited by a non-negative integer

    Arguments:
    L: list or tuple of int
        List or tuple containing the maximum integer value allowed for each dimension of the lists to generate.
    """

    # Note: documentation on the `yield` expression: https://docs.python.org/3/reference/expressions.html#yieldexpr

    def all_combos_with_max_limits_in_sublist(v, dim):
        r = dim
        if r < R - 1:
            # Set the current dimension value to all its possible values
            # and for each of those values solve the sub-problem of finding all valid combinations
            # for the sub-list that is to the RIGHT of r (this step calls this same function recursively)
            for k in range(L[dim] + 1):
                v[r] = k
                # NOTE THE USE OF `yield from` in order to get an yield value from a recursive call!
                yield from all_combos_with_max_limits_in_sublist(v, r + 1)
        else:
            # No degrees of freedom left for the last dimension (right-most index)
            # => set its value to all its possible values accepted by the corresponding dimension of L
            for k in range(L[dim] + 1):
                v[r] = k
                yield v

    if not isinstance(L, (list, tuple)):
        raise ValueError("Parameter `L` must be a list: {}".format(L))
    if not all([is_integer(l) and l >= 0 for l in L]):
        raise ValueError("The values in parameter `L` must be non-negative integers: {}".format(L))

    R = len(L)
    # Initialize the list on the combination that is valid for sure (i.e. list with all zeros, as the limits in L must be at least 0)
    v = [0] * R
    # We start the combination generation by calling "the generator that solves every sub-problem" on the WHOLE list (i.e. on the list with its full dimension)
    gen = all_combos_with_max_limits_in_sublist(v, 0)
    return gen


def compute_set_of_frequent_states(states, threshold=0.05, cumulative=False):
    dist_state_counts = pd.Series(states).value_counts(normalize=True)
    if cumulative:
        cum_dist_state_counts = np.cumsum(dist_state_counts)
        return set(cum_dist_state_counts.index[cum_dist_state_counts <= threshold])
    else:
        return set(dist_state_counts.index[dist_state_counts > threshold])


def compute_set_of_frequent_states_with_zero_reward(states, rewards, threshold=0.05, cumulative=False):
    # Convert to series
    if len(states) != len(rewards):
        raise ValueError(f"The number of elements in `states` and the number of elements in `rewards` must be the same: {len(states)}, {len(rewards)}")
    states = pd.Series(states)
    rewards = pd.Series(rewards)

    # Filter on states with zero reward
    ind_zero_reward = rewards == 0
    n_nonzero_rewards = sum(ind_zero_reward)

    dist_state_counts = pd.Series(states[ind_zero_reward]).value_counts(normalize=True)

    if cumulative:
        cum_dist_state_counts = np.cumsum(dist_state_counts)
        return set(cum_dist_state_counts.index[cum_dist_state_counts <= threshold])
    else:
        return set(dist_state_counts.index[dist_state_counts > threshold])


def compute_transition_matrices(env, policy):
    """
    Computes the transition probability matrices for the EPISODIC and CONTINUING learning tasks on a given discrete-state / discrete-action environment
    where an agent acts under the given policy with transition probabilities defined by the environment (i.e. the environment defines the probability
    of going from each state s to any other state in the environment when taking each possible action `a` accepted by the environment at state s.

    For the CONTINUING learning task, when a terminal state is reached, the agent is assumed to start at a state chosen following
    the initial state distribution defined in the environment.

    Arguments:
    env: EnvironmentDiscrete
        Environment with discrete states and discrete actions with ANY initial state distribution.
        Rewards can be anywhere.

    policy: policy object with method getPolicyForAction(a, s) defined, returning Prob(action | state)
        Policy object acting on a discrete-state / discrete-action environment.
        This could be of class e.g. probabilistic.PolGenericDiscrete or PolNN.
        Normally the policy for all states can be retrieved by either policy.getPolicy() (as a dictionary) or policy.get_policy_values() (as a matrix whose
        rows are indexed by the states.

    Return: tuple
    Tuple with the following 6 elements:
    - P_epi: matrix containing the transition probability matrix for the EPISODIC learning task.
    - P_con: matrix containing the transition probability matrix for the CONTINUING learning task where, once the Markov chain reaches
    a terminal state, it restarts at an environment's start state chosen following the environment's initial state distribution.
    - b_epi: 1D array containing the expected reward over all actions for each state in the system under the EPISODIC learning task.
    - b_con: 1D array containing the expected reward over all actions for each state in the system under the CONTINUING learning task.
    - g: expected reward over all states under stationarity for the CONTINUING learning task.
    - mu: the stationary probability under the CONTINUING learning task.
    """
    nS = env.getNumStates()

    # Transition probability matrix under the given policy for the EPISODIC learning task
    # It is based on the environment transition probabilities (given the action at each state) and the defined probabilistic policy
    # i.e. P[x,y] = sum_{a}{ p(x,y|a) * policy(a|x) }, where p(x,y|a) is the transition probability of going from x -> y when taking action a at state x.
    P = np.matrix(np.zeros((nS, nS)))
    for s in range(nS):
        for a in range(env.getNumActions()):
            # In the environment, P[s][a] is a list of tuples of the form (prob, next_state, reward, is_terminal) (see the DiscreteEnv environment defined in envs/toy_text/discrete.py)
            # and in each tuple, the transition probability to each possible next state for the GIVEN action 'a' is given at index 0 and the next state is given at index 1.
            for info_transition in env.P[s][a]:
                # Next state to be updated in the transition probability matrix
                ns = info_transition[1]
                prob_transition_from_s_to_ns_given_action_a = info_transition[0]
                prob_taking_action_a_at_state_s = policy.getPolicyForAction(a, s)
                P[s, ns] += prob_taking_action_a_at_state_s * prob_transition_from_s_to_ns_given_action_a

    # Transition matrix for the EPISODIC learning task (where the rightmost state is TERMINAL)
    P_epi = P

    # Transition matrix for the CONTINUING learning task where, when reaching a terminal state, the process restarts at an environment's start state following
    # its initial state distribution.
    P_con = P.copy()
    for s in env.getTerminalStates():
        P_con[s, :] = env.getInitialStateDistribution()

    # Check that P_epi and P_con are valid transition probability matrices
    for s in range(nS):
        assert np.isclose(np.sum(P_epi[s]), 1.0), f"Row P[s={s}] for the EPISODIC environment must sum up to 1: sum={np.sum(P_epi[s])}"
        assert np.isclose(np.sum(P_con[s]), 1.0), f"Row P[s={s}] for the CONTINUING environment must sum up to 1: sum={np.sum(P_con[s])}"

    # Stationary probability distribution of the Markov chain associated to the CONTINUING learning task, which is needed to compute the expected or average reward or bias g below
    eigenvalues, eigenvectors = np.linalg.eig(P_con.T)
    idx_eigenvalue_one = np.where(np.abs(eigenvalues - 1.0) < 1E-6)[0][0]
    assert np.isclose(eigenvalues[idx_eigenvalue_one], 1.0)
    eigenvector_one = eigenvectors[:, idx_eigenvalue_one]
    mu = np.squeeze(np.array(np.abs(eigenvector_one) / np.sum(np.abs(eigenvector_one))))

    # Independent terms `b` of the `(I - P)*V = b - g*1` Bellman equation, for the EPISODIC (where g = 0) and the CONTINUING learning tasks
    b_epi = np.array([np.sum([P_epi[x, y] * env.getReward(y) for y in range(nS)]) for x in range(nS)])
    b_con = np.array([np.sum([P_con[x, y] * env.getReward(y) for y in range(nS)]) for x in range(nS)])

    # Expected reward, i.e. the average reward observed over all states under stationarity for the CONTINUING task, since avg.reward = sum_{x} mu[x] * r[x]
    # Recall that the average reward is the one that makes the system of equations satisfied by V(s) (the Bellman equations) feasible (i.e. consistent, as opposed to inconsistent).
    # And recall that the average reward g appears in the independent term of the Bellman equations which is `b - g*1`.
    # For more details, see the very good notes by Ger Koole: https://www.scribd.com/document/177118281/Lecture-Notes-Stochastic-Optimization-Koole
    g = sum([mu[x] * env.getReward(x) for x in range(nS)])

    return P_epi, P_con, b_epi, b_con, g, mu


def compute_state_value_function_from_transition_matrix(P, expected_one_step_reward, bias=0.0, gamma=1.0):
    """
    Computes the state value function V(s) based on the transition matrix, the expected one-step reward (b) (expectation computed over all possible actions),
    bias (g, a.k.a. expected reward under stationarity which is needed for the average reward criterion) and discount factor (gamma)

    To avoid problems with over parameterization of system, we compute the minimum norm inverse of (I - gamma*P), which does not have an inverse when gamma = 1.

    Arguments:
    P: np.matrix
        Transition probability matrix of the Markov chain whose state value function is of interest.

    expected_one_step_reward: np.array
        Array containing the expected one-step reward (over all actions) for each state of the Markov chain.
        It's the `b` term in the `(I - gamma*P) * V = b - g*1` Bellman equation.

    bias: (opt) float
        Expected reward over all states under stationarity, i.e. g = sum_{x} mu[x] * r[x].
        Recall that the expected reward is the one that makes the system of equations satisfied by V(s) (the Bellman equations)
        feasible (i.e. consistent, as opposed to inconsistent).
        This value should be given only under the CONTINUING learning task.
        For more details, see the very good notes by Ger Koole: https://www.scribd.com/document/177118281/Lecture-Notes-Stochastic-Optimization-Koole
        default: 0.

    gamma: (opt) float in (0, 1]
        Discount factor for the DISCOUNTED learning criterion.
        default: 1.0
    """
    b = expected_one_step_reward
    g = bias
    V = np.asarray(np.dot(np.linalg.pinv(np.eye(len(P)) - gamma*P), b - g))[0]
    return V


def generate_min_exponential_time(rates):
    """
    Generates a realization of the minimum of exponential times at the given rates

    Arguments:
    rates: positive float or list of positive floats or numpy array of positive floats
        Rates of the exponential distributions on which the minimum time is generated.
        Some (but not all) of the rates may be NaN as only the non-NaN values are considered for the possible
        exponential distributions.

    Return: tuple
    Tuple with the following elements:
    - time: realization of the minimum time among the exponential distributions of the given rates.
    - idx: index indicating which exponential rate the generated time should be associated with, which is chosen
    randomly out of all valid rates given (where a valid rate means that it's a non-negative real number).
    """
    # Check the rates
    rates = as_array(rates)
    if any(rates[~np.isnan(rates)] < 0):
        warnings.warn("Some of the rates are negative... they will be ignored for the generation of the min exponential time ({}".format(rates))

    # Find the valid rates
    valid_rates = copy.deepcopy(rates)
    is_valid_rate = np.array([True if r > 0 else False for r in rates])
    valid_rates[~is_valid_rate] = np.nan

    # Rate of occurrence of ANY event among the input rates
    event_rate = np.nansum(valid_rates)
    if event_rate <= 0:   # NOTE that, in case all valid_rates are NaN, the nansum() of all NaNs is 0.0! (so this condition still takes care of the all-NaN-rates case)
        raise ValueError("The event rate computed from the given rates ({}) must be positive ({})".format(rates, event_rate))

    # Generate the event time
    event_time = np.random.exponential(1/event_rate)

    # Probability of selection of each possible event with a valid rate
    probs = valid_rates[is_valid_rate] / event_rate
    indices_to_choose_from = [idx for idx, rate in enumerate(valid_rates) if not np.isnan(rate)]
    #print("probs: {}".format(probs))   # For 5 particles with rates lambda = 0.7, mu = 1.0 each, these probabilities are 0.082353 (for each of the 5 lambda's) and 0.11765 (for each of the 5 mu's)
    #print("indices to choose from: {}".format(indices_to_choose_from))

    # Define the index on which the generated event time should be associated with
    idx_event = np.random.choice(indices_to_choose_from, size=1, p=probs)[0]    # Need `[0]` because the value returned by random.choice() is an array

    return event_time, idx_event


def compute_survival_probability(survival_times: Union[list, deque], colnames: list=None, right_continuous=True):
    """
    Computes the survival probability from a list of sorted survival times

    Arguments:
    survival_times: list or deque
        List containing the observed survival times on which the step survival probability is computed.
        The list is assumed SORTED by increasing times, but this is NOT checked because it takes time.

    colnames: (opt) list or array-like of length 2
        Column names to be used for the survival times and the survival probability, respectively.
        default: ['t', 'P(T>t)'] when right_continuous=True, ['t', 'P(T>=t)'] when right_continuous=False

    right_continuous: (opt) bool
        Whether the computed survival probability is right-continuous (True), i.e. it corresponds to the probability function P(T>t)
        or left-continuous (False), i.e. it corresponds to the probability function P(T>=t).
        default: True

    Return: pandas DataFrame
    Data frame containing the following two columns:
    - `colnames[0]`: the input survival times (assumed sorted)
    - 'colnames[1]': the survival probability for the corresponding survival time value, which corresponds to a piecewise constant estimation
    of the survival probability. If right_continuous=True, the value is associated to the corresponding t value and to the time interval that is
    to the RIGHT of it, making t the LOWER bound of such interval. If right_continuous=False, the value is associated to the corresponding t value
    and to the time interval that is to the LEFT of it, making t the UPPER bound of such interval.

    Note that the data frame has as many rows as the length of `survival_times` where the first row has time value t = 0.0.
    When right_continuous=True, the first row has value 1.0 for P(T>t), which is the value of the survival probability function at t = 0.0,
    as the function is right continuous.
    When right_continuous=False, the first row has value 0.0 for P(T>=t) so that, if that row is used in the calculation of an integral / sum,
    it does NOT contribute to it, as it should be the case, since the value t=0 corresponds to the right-end of the interval associated to the value
    stored for P(T>=t). Note that we choose to have that record in the output data frame so that a call to utils.basic.merge_values_in_time() works fine
    as this function requires the first value of each times list to merge to be always 0.
    """
    if colnames is None:
        colnames = ['t', 'P(T>t)'] if right_continuous else ['t', 'P(T>=t)']
    elif not isinstance(colnames, (list, tuple, np.ndarray)) or len(colnames) != 2:
        raise ValueError(f"Input parameter `colnames` must be either list, tuple or array and its length must be 2 ({colnames})")

    if not isinstance(survival_times, (list, deque)):
        raise ValueError("The `survival_times` parameter must be of type list")
    if len(survival_times) == 0 or survival_times[0] != 0.0:
        raise ValueError("The `survival_times` parameter must have at least one element and the first element must be 0.0")

    # Number of observed death events used to measure survival times
    N = len(survival_times) - 1

    if N > 0:
        proba_surv = [n / N for n in
                      range(N, -1*right_continuous, -1)]  # This is N down-to 0 if right_continuous=True, o.w. it's N down-to 1
        if right_continuous:
            assert proba_surv[-1] == 0.0
    else:
        proba_surv = [1.0] if right_continuous else [1.0]

    assert proba_surv[0] == 1.0

    return pd.DataFrame.from_items([(colnames[0], survival_times),   # We remove the initial value already included in survival_times, t=0.0, if the survival probability is requested to be LEFT continuous.
                                    (colnames[1], [0.0] + proba_surv if not right_continuous else proba_surv)])


def get_server_loads(job_rates, service_rates):
    """
    Returns the server loads (rhos) for a queue system from the job arrival rates and the service rates"

    Arguments:
    job_rates: list or array
        Arrival rates of the jobs by server.

    service_rates: list or array
        Server service rates.

    Return: list
    List of server loads, rhos computed as lambda / mu, where lambda is the job arrival rate and mu is the service rate
    of each server.
    """
    return [b/d for b, d in zip(job_rates, service_rates)]


def compute_job_rates_by_server(job_class_rates: Union[list, tuple, np.ndarray], nservers, policy_assign_map):
    """
    Computes the equivalent job arrival rates for each server from the job arrival rates (to the single buffer)
    and the job assignment policy of the agent.

    This can be used when the job is pre-assigned to a server at the moment of arrival (as opposed to being
    assigned when a server queue is freed).

    Arguments:
    job_class_rates: list or array
        Arrival rate of each job class to the queue buffer.

    nservers: int
        Number of servers in the queue system.

    policy_assign_map: list of lists
        List of probabilities of assigning each job class to a server in the queue system.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

    Return: list
    The equivalent job arrival rates for each server r, computed as:
    job_arrival_rate[r] = sum_{c over job classes} { job_rate[c] * Pr(assign job c to server r) }
    """
    R = nservers
    J = len(job_class_rates)
    job_rates_by_server = [0]*R
    for r in range(R):
        for c in range(J):
            job_rates_by_server[r] += policy_assign_map[c][r] * job_class_rates[c]

    return job_rates_by_server


def compute_number_of_burnin_cycles_from_burnin_time(cycle_times, burnin_time):
    """
    Computes the number of burn-in cycles to remove from the beginning of a simulation
    when computing expectations in order to guarantee a specified burn-in continuous time
    """
    if burnin_time > 0:
        # Find the number of cycles to remove from the beginning of the simulation in order to guarantee that
        # the time at which the first cycle considered starts is larger than the burn-in time
        # The logic is best illustrated by an example:
        # burnin_time = 3.5
        # cycle_times = [0.3, 0.7, 2.8, 0.4, 0.9]
        # => cumsum(cycle_times) = [0.3, 1.0, 3.8, 4.2, 5.1]
        #   => the loop below stops at idx = 2 (because t=3.8 >= burnin_time=3.5)
        #       => burnin_cycles = 3 (that is we need to remove the first 3 cycles --including the third one--
        #                              because all those cycles have a non-empty intersection with the interval
        #                              [0, burnin_time=3.5]. So, the first cycle that is falls FULLY in the
        #                              period assumed stationary is the fourth cycle that goes from 3.8 thru 4.2
        #                              whose cycle time is 0.4, i.e. taken from index 3 of the cycle_times list)
        assert cycle_times is not None
        burnin_cycles = len(cycle_times)  # Initialize the number of burn-in cycles to the maximum possible (where all cycles are removed actually!)
        for idx, t in enumerate(np.cumsum(cycle_times)):
            if t >= burnin_time:
                burnin_cycles = idx + 1  # See example above to justify why we sum +1 to idx to get the number of burn-in cycles
                break
    else:
        burnin_cycles = 0

    return burnin_cycles


def compute_blocking_probability_birth_death_process(rhos: list, capacity: int):
    """
    Computes the true blocking probability of a birth-death process with R servers and total capacity C.
    
    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    Return: float
    The probability that the process is at its max capacity.
    """
    if not isinstance(rhos, list) or len(rhos) == 0:
        raise ValueError("Input parameter `rhos` must be a non-empty list: {}".format(rhos))
         
    C = capacity
    x, dist = stationary_distribution_product_form(C, rhos, func_prod_birthdeath)
    proba_blocking = 0.0
    for xx, dd in zip(x, dist):
        if np.sum(xx) == C:
            proba_blocking += dd

    return proba_blocking


def compute_blocking_probability_knapsack(capacity: int, rhos: list, lambdas: list, blocking_sizes: list=None):
    """
    Computes the probability of blocking an incoming job (of any class) to a stochastic knapsack of a given capacity

    Arguments:
    capacity: int
        Capacity of the knapsack, independently of the blocking sizes by job class.

    rhos: list
        Load of each job class arriving to the knapsack.

    lambdas: list
        Arrival rate of each job class.

    blocking_sizes: (opt) list
        Sizes (occupations) of the different job classes in the knapsack at which an incoming job of the respective class is blocked.
        default: None, in which case the full capacity of the knapsack is used to compute the stationary probability distribution
        and no limitation to the occupancy of each job class is applied.

    Return: float
    The probability of blocking an incoming job (of ANY class).
    """
    if  not isinstance(rhos, (list)) or \
        not isinstance(lambdas, (list)):
        raise ValueError("Parameters `rhos`, `lambdas` must be of type list")
    if  len(rhos) == 0 or \
        len(lambdas) == 0:
        raise ValueError("Input parameter `rhos` and `lambdas` must be a non-empty list: {}, {}".format(rhos, lambdas))
    if not len(rhos) == len(lambdas):
        raise ValueError("Parameters `rhos` and `lambdas` must have the same length: {}, {}".format(len(rhos), len(lambdas)))

    effective_capacity, states, probas_stationary = \
        compute_stationary_probability_knapsack_when_blocking_by_class(capacity, rhos, blocking_sizes=blocking_sizes)

    if blocking_sizes is None:
        # If no blocking sizes are given, each class is blocked at the knapsack's capacity (e.g. [10, 10, 10])
        blocking_sizes = [capacity] * len(rhos)
        assert effective_capacity == capacity

    dict_probas_stationary = dict([[tuple(x), d] for x, d in zip(states, probas_stationary)])
    proba_blocking = compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates(dict_probas_stationary,
                                                                                                    effective_capacity,
                                                                                                    lambdas,
                                                                                                    blocking_sizes)

    return proba_blocking


def compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates( probas_stationary: dict,
                                                                                    capacity: int,
                                                                                    lambdas: list,
                                                                                    blocking_sizes: list):
    if not isinstance(probas_stationary, dict):
        raise ValueError("Parameter `probas_stationary` must be dictionary: {}".format(probas_stationary))
    if not isinstance(lambdas, list) or not isinstance(blocking_sizes, list) or len(lambdas) != len(blocking_sizes):
        raise ValueError("Parameter `lambdas` and `blocking_sizes` must be list and have the same length: lambdas={}, blocking_sizes={}".format(lambdas, blocking_sizes))

    proba_blocking = 0.0
    Lambda = np.sum(lambdas)
    for x, p in probas_stationary.items():
        ## Note: the stationary probabilities reported in probas_stationary do NOT need to be just the probabilities
        ## for the blocking states, but they should include them. In fact, the blocking condition is defined below
        ## when computing the contribution from each job class arrival.
        total_x = np.sum(x)

        # Before multiplying by the stationary probability,
        # we first compute the contribution from each job class to blocking because this depends on whether just one class
        # can block or more than one class can block, or whether ALL classes block (which happens when the system is at full capacity)
        contribution_from_jobclasses = 0.0
        for j, lambdaj in enumerate(lambdas):
            contribution_from_jobclasses += lambdaj if total_x == capacity or x[j] == blocking_sizes[j] \
                                            else 0.0

        proba_blocking += p * contribution_from_jobclasses
    proba_blocking /= Lambda

    return proba_blocking


def stationary_distribution_product_form(capacity: int, rhos: list, func_prod):
    """
    Computes the stationary distribution on all possible states of a system having product form distribution
    for holding a set of items with load `rhos` up to a given capacity.

    Arguments:
    capacity: int
        The capacity of the system.

    rhos: list
        The load of each item class accepted in the system (load = arrival-rate / service-rate).

    func_prod: Callable
        Function that should be used to compute the individual contribution of the occupancy of each item class
        to the product form distribution of a particular occupancy state of all item classes n = (n1, n2, ..., nR),
        where R is the number of classes, e.g. for n = (3, 2, 5).
        For more details see the documentation for `stationary_distribution_product_form_fixed_occupancy_unnormalized()`.

    Return: tuple
    The tuple contains the following elements:
    - a list of each possible state n = (n1, n2, ..., nR) such that the sum of the n(j)'s is less than or equal to the given capacity.
    - a list with the stationary probability of occurrence of each state given in the first list.
    """
    if capacity < 0:
        raise ValueError("The capacity must be non-negative: {}".format(capacity))

        # Create short-name variable for number of servers
    R = len(rhos)

    # Total expected number of cases (needed to initialize the output arrays and check the results)
    ncases_total_expected = 0
    for c in range(capacity + 1):
        ncases_total_expected += comb(c + R - 1, c)

    # Initialize the output arrays with all possible x = (n1, n2, ..., nR) combinations,
    # and the probability distribution for each x. They are all initialized to a dummy value.
    # They are indexed by the order in which all the combinations are generated below:
    # - by increasing capacity 0 <= c <= capacity
    # - by the order defined by the all_combos_with_sum(R,c) function
    x = [[-1] * R] * ncases_total_expected
    dist = [0] * ncases_total_expected
    ncases_total = 0
    const = 0  # Normalizing constant (because the system's capacity is finite)
    last_case = -1
    for c in range(capacity + 1):
        ncases_expected = comb(c + R - 1, c)
        ind = slice(last_case + 1, last_case + 1 + ncases_expected)

        x[ind], dist[ind] = stationary_distribution_product_form_fixed_occupancy_unnormalized(c, rhos, func_prod, ncases_expected)

        const += sum(dist[ind])
        ncases_total += ncases_expected
        # Prepare for next iteration
        last_case += ncases_expected
    dist /= const
    assert ncases_total == ncases_total_expected, \
        "The number of TOTAL generated combinations for R={}, C<={} ({}) must be equal to the expected number of combinations ({})" \
        .format(R, C, ncases_total, ncases_total_expected)
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function must be equal to 1 ({:.6f})".format(sum(dist))

    return x, dist


def stationary_distribution_product_form_fixed_occupancy_unnormalized(occupancy: int, rhos: list, func_prod, ncases_expected=None):
    """
    Computes the UNNORMALIZED stationary distribution on a system having product form distribution for the state subspace
    associated to a fixed total occupancy, i.e. for all the n = (n1, n2, ..., nR) such that their sum is equal to
    the given occupancy.

    This function is used as a helper function for the calculation of the stationary distribution of the whole
    state space of a product form stationary distribution, where the components of n sum up to all occupancies that
    do not overcome the system's capacity.

    Arguments:
    capacity: int
        The capacity of the system.

    rhos: list
        The load of each item class accepted in the system (load = arrival-rate / service-rate).

    func_prod: Callable
        Function that should be used to compute the individual contribution of the occupancy of each item class
        to the product form distribution of a particular occupancy state of all item classes n = (n1, n2, ..., nR),
        where R is the number of classes.
        The signature of the function should be the following:
            func(rho, n)
        where
        - rho is the load of the class of interest.
        - n is the occupancy in the system of the class of interest.

    ncases_expected: (opt) int
        The expected number of states satisfying the occupancy condition mentioned in the description.
        This number may have been computed elsewhere and it is used here if given, in order to avoid
        its recalculation.

    Return: tuple
    The tuple contains the following elements:
    - a list containing each possible state n = (n1, n2, ..., nR) such that the sum of the n(j)'s is equal to the given occupancy.
    - a list with the unnormalized probability of occurrence of each state given in the first list.
    In order to normalize it to a distribution, each value should be divided by the sum of all the values in the list.
    """
    R = len(rhos)
    C = occupancy
    if ncases_expected is None:
        ncases_expected = comb(C + R - 1, C)
    combos_generator = all_combos_with_sum(R, C)
    ncases = 0
    x = [[-1] * R] * ncases_expected
    dist = [0] * ncases_expected
    while True:
        try:
            next_combo = next(combos_generator)
            # print("next_combo (k={}): {}".format(ncases, next_combo))
            x[ncases] = copy.deepcopy(next_combo)  # IMPORTANT: Need to make a copy o.w. x[k] will share the same memory address as next_combo and its value will change at the next iteration!!
            dist[ncases] = np.prod([func_prod(r, nr) for r, nr in zip(rhos, next_combo)])    # Note: next_combo is a tuple containing the number of jobs of each class in the system: (n1, n2, ..., nR)
            ncases += 1
        except StopIteration:
            break
    combos_generator.close()
    assert ncases == ncases_expected, \
        "The number of generated combinations for R={}, C={} ({}) is equal to the expected number of combinations ({})" \
        .format(R, C, ncases, ncases_expected)

    return x, dist


def compute_expected_cost_knapsack(costs: list, capacity: int, rhos: list, lambdas: list):
    """
    Computes the expected cost of a stochastic knapsack receiving multi-class jobs having potentially different costs of blocking

    Arguments:
    costs: list
        Costs of blocking each job class.

    capacity: int
        Number of servers in the knapsack.

    rhos: list
        Load of each job class.

    lambdas: list
        Arrival rate of each job class.

    Return: dict
    Dictionary containing each possible set of blocking sizes (job occupancies) as keys and their expected cost
    of blocking a job (of any class) arriving to the stochastic knapsack, as value.
    """
    if  not isinstance(costs, (list)) or \
        not isinstance(rhos, (list)) or \
        not isinstance(lambdas, (list)):
        raise ValueError("Parameters `costs`, `rhos`, `lambdas` must be of type list")
    if not len(costs) == len(rhos) and len(rhos) == len(lambdas):
        raise ValueError("Parameters `costs`, `rhos`, `lambdas` must all have the same length: {}, {}, {}".format(len(costs), len(rhos), len(lambdas)))

    if capacity <= 3:
        printFlag = True
    else:
        printFlag = False

    # All possible blocking sizes to consider: these are all states (x1, x2, ..., xR) that sum up to C <= R*capacity while satisfying that x(j) <= capacity
    # Ex: if R = 3 and capacity = 10, states that satisfy these conditions are e.g. (10, 10, 10), (10, 8, 0),
    # where we see that in both cases the sum of the x values exceed the system's capacity 10 but do NOT exceed R*capacity = 30,
    # and the following states do NOT satisfy the condition (10, 13, 0), (15, 10, 3),
    # although in both cases the sum of the x values are still smaller than R*capacity = 30;
    # however in these cases the states do not qualify because at least one of the components x(j) is > capacity,
    # meaning that blocking at that value is the same as blocking at x(j) = capacity, therefore this state
    # does NOT *need* to be considered (although it could).
    # Note that it's important to consider these larger set of blocking sizes (i.e. larger than the set of all possible states where the system could be in)
    # because they are valid blocking sizes to be set for EACH job class. I.e. the blocking sizes do NOT have to be valid system states, they have
    # to satisfy the univariate condition that their components are <= capacity of the knapsack.
    all_blocking_sizes = []
    R = len(rhos)
    for C in range(R*capacity + 1):
        combos_generator = all_combos_with_sum(R, C)
        while True:
            try:
                combo = next(combos_generator)
                if all(np.array(combo) <= capacity):
                    all_blocking_sizes += [tuple(combo)]
            except StopIteration:
                break

    # Iterate on all possible blocking sizes and compute the expected cost for each of them
    expected_costs = dict()
    Lambda = np.sum(lambdas)
    for blocking_sizes in all_blocking_sizes:
        # Compute the stationary probabilities for the valid states x in which the Markov chain can be,
        # *given* the currently considered blocking sizes
        # (a valid state x must satisfy that each state dimension x(j) <= blocking_size(j))
        # by renormalizing the stationary probability of a knapsack having capacity equal to
        # the sum of blocking sizes on the valid states.
        capacity_blocking, states_valid_when_blocking, probas_stationary_when_blocking = \
            compute_stationary_probability_knapsack_when_blocking_by_class(capacity, rhos, blocking_sizes)

        if printFlag:
            print("\n------- ", blocking_sizes)
        expected_costs[blocking_sizes] = 0.0
        for idx, x in enumerate(states_valid_when_blocking):
            total_x = np.sum(x)
            if printFlag:
                print(x, total_x, capacity_blocking, "p=", probas_stationary_when_blocking[idx], end=":    ")
            # Cost of leaving unused servers in the system: number of unused servers, imputed even if there is no blocking
            #cost_unused_servers = max(0, capacity - total_x)
            cost_unused_servers = 0
            contribution_from_jobclasses = 0.0
            for j, costj in enumerate(costs):
                contribution_from_jobclasses +=  costj * lambdas[j] if total_x == capacity_blocking or x[j] == blocking_sizes[j] \
                                                                    else 0.0
            if printFlag:
                print("cost unused = {:.3f}".format(cost_unused_servers), end=", ")
                print("cost*lambda = {:.3f}".format(contribution_from_jobclasses), end=" --> ")
            expected_costs[blocking_sizes] += probas_stationary_when_blocking[idx] * (cost_unused_servers + contribution_from_jobclasses)
            if printFlag:
                print("Expected cost = {:.3g}".format(expected_costs[blocking_sizes]))
        expected_costs[blocking_sizes] /= Lambda
        if printFlag:
            print("-----------> ", expected_costs[blocking_sizes])

    return expected_costs


def compute_stationary_probability_knapsack_when_blocking_by_class(capacity: int, rhos: list, blocking_sizes: list=None):
    """
    Computes the stationary probability of all states in a knapsack with a given capacity that is potentially limited by
    a policy that blocks incoming jobs at the given blocking sizes by job class

    capacity: int
        Capacity of the knapsack, independently of the blocking sizes by job class.

    rhos: list
        Load of each job class arriving to the knapsack.

    blocking_sizes: (opt) list or tuple or numpy.ndarray
        Sizes (occupations) of the different job classes in the knapsack at which an incoming job of the respective class is blocked.
        default: None, in which case the full capacity of the knapsack is used to compute the stationary probability distribution
        and no limitation to the occupancy of each job class is applied.

    Return: tuple
    - capacity_blocking: the capacity of the knapsack taking into account the blocking sizes.
    - states_valid_when_blocking: list of states that are valid in the knapsack whose capacity is potentially limited
    by the blocking sizes of the different job classes, and each job class occupancy is limited by the respective blocking size.
    - probas_stationary_when_blocking: the stationary probability of each valid state in the knapsack whose capacity is potentially limited by the
    blocking sizes of the different job classes, and each job class occupancy is limited by the respective blocking size.
    """
    if blocking_sizes is None:
        blocking_sizes = [capacity] * len(rhos)

    capacity_blocking = min(capacity, np.sum(blocking_sizes))  # e.g. min(10, sum([2, 1, 0])) = 3 or min(10, sum([8, 8, 6]) = 10
    states, p = stationary_distribution_product_form(capacity_blocking, rhos, func_prod_knapsack)
    states_valid_when_blocking, probas_stationary_when_blocking = adjust_stationary_probability_knapsack_when_blocking(states, p, blocking_sizes)

    return capacity_blocking, states_valid_when_blocking, probas_stationary_when_blocking


def adjust_stationary_probability_knapsack_when_blocking(states, p, blocking_sizes):
    """
    Adjusts the stationary probabilities of a knapsack (computed theoretically or estimated by simulation) with a given capacity
    that is potentially limited by a policy that blocks incoming jobs at the given blocking sizes by job class.

    Note that the knapsack's capacity is not a parameter of this function, but it can be inferred from the `states` parameter.

    states: list
        List of states on which the stationary probability has been computed.
        Some of these states may NOT be valid states given the blocking sizes.

    p: list
        List of the stationary probability computed or estimated for each state in `states` that need to be adjusted.
        If the stationary probability of a state is NaN, it is assumed it is zero.

    blocking_sizes: list or tuple or numpy.ndarray
        Sizes (occupations) of the different job classes in the knapsack at which an incoming job of the respective class is blocked.
        default: None, in which case the full capacity of the knapsack is used to compute the stationary probability distribution

    Return: tuple
    - states_valid_when_blocking: list of states that are valid in the knapsack whose capacity is potentially limited
    by the blocking sizes of the different job classes, and each job class occupancy is limited by the respective blocking size.
    - p: the stationary probability of each valid state in the knapsack whose capacity is potentially limited by the
    blocking sizes of the different job classes, and each job class occupancy is limited by the respective blocking size.
    """
    states_valid_when_blocking = []  # List of states that are valid when blocking at the current blocking_sizes
    probas_stationary_when_blocking = []
    for idx, x in enumerate(states):
        if all(np.array(x) <= blocking_sizes):
            # We have filtered here on valid states, i.e. for which x(j) <= blocking_sizes(j) for each job class j
            # e.g. if blocking_sizes = [2, 1, 0] (i.e. the maximum knapsack's capacity is 3),
            # then e.g. [0, 1, 1] is NOT a valid state, despite being a possible state for a knapsack with capacity 3,
            # because x[2] = 1 > blocking_sizes[2] = 0.
            states_valid_when_blocking += [x]
            probas_stationary_when_blocking += [p[idx] if not np.isnan(p[idx]) else 0.0]
    # Normalize the stationary probabilities of the valid states
    p_blocking_sum = np.sum(probas_stationary_when_blocking)
    if p_blocking_sum > 0:
        probas_stationary_when_blocking = [pp / p_blocking_sum for pp in probas_stationary_when_blocking]
    assert np.sum(probas_stationary_when_blocking) == 0.0 or np.isclose(np.sum(probas_stationary_when_blocking), 1.0), \
        f"Either all adjusted probabilities are 0 or their sum is equal to 1: sum(probas_stationary_when_blocking) = {np.sum(probas_stationary_when_blocking)}"

    return states_valid_when_blocking, probas_stationary_when_blocking


def func_prod_birthdeath(rho, n):
    return (1 - rho) * rho ** n


def func_prod_knapsack(rho, n):
    # What I found about the knapsack in Python: https://www.pythonpool.com/knapsack-problem-python/, which presents 3 methods to SOLVE the kanpsack problem (i.e. maximize the number of items based on their values and required resources)
    # Searching for 'python function to compute stationary distribution of knapsack'
    # I haven't found a function that computes the stationary distribution.
    return rho ** n / factorial(n)


# Tests
if __name__ == "__main__":
    import pytest   # For pytest.raises() which allows to check if an exception is raised when calling a function

    #----------------- rmse() and mape() --------------------#
    print("\n--- Testing rmse() and mape():")

    # Test of one difference 0.0 with true value 0.0. The APE for that value should be 0.0 and the MAPE should NOT be NaN
    x = np.array([3, 2, 0.0, -1.0])
    assert np.abs(rmse(2*x, x) - 1.8708287) < 1E-6
    assert np.abs(mape(2*x, x) - 0.375) < 1E-6

    # Test that when there is at least one division by 0, the MAPE yields Inf
    x = np.array([3, 2, 0.0, -1.0])
    assert np.abs(rmse(x, x + 1) - 1.0) < 1E-6
    assert mape(x, x + 1) == np.Inf

    # Test that when there is at least one NaN the RMSE and MAPE yield NaN
    x = np.array([3, 2, 0.0, np.nan])
    y = np.array([3, 5.1, 0.0, 2.1])
    assert np.isnan(rmse(x, y))
    assert np.isnan(mape(x, y))
    assert np.isnan(rmse(y, x))
    assert np.isnan(mape(y, x))

    # Test 2D arrays
    # These results were verified in Excel
    x_2d = np.array([[3, 2, 0.0, -1.0],
                     [4, 8, -9.1, 4.0]])
    y_2d = np.array([[3.2, 1.8, 0.0, 4.5],
                    [4.5, 6.8, -5.4, 2.1]])
    assert np.abs(rmse(x_2d, y_2d) - 2.4829418) < 1E-6
    assert np.abs(mape(x_2d, y_2d) - 0.8529075) < 1E-6

    # Test weighted versions
    # These results were verified in Excel
    weights_2d = np.array([[1.0, 3.2, 0.5, 0.0],
                           [2.1, 5.1, 0.8, 0.7]])
    assert np.abs(rmse(x_2d, y_2d, weights=weights_2d) - 1.2671509) < 1E-6
    assert np.abs(mape(x_2d, y_2d, weights=weights_2d) - 0.1546224) < 1E-6

    # Test error when different shapes given
    # Note that in order to use unittest.TestCase.assertRaises() we need to create an INSTANCE of the TestCase class
    # Otherwise we get the error "assertRaises() arg 1 must be an exception type or tuple of exception types"
    # Ref: https://stackoverflow.com/questions/18084476/is-there-a-way-to-use-python-unit-test-assertions-outside-of-a-testcase
    # which was referenced by Martijn Pieters in https://stackoverflow.com/questions/49369163/custom-exceptions-in-unittests
    x_2d = np.array([[3, 2, 0.0],
                     [4, 8, -9.1]])
    y_2d = np.array([[3.2, 1.8, 0.0, 4.5],
                    [4.5, 6.8, -5.4, 2.1]])
    tc = TestCase()
    tc.assertRaises(ValueError, rmse, x_2d, y_2d)
    tc.assertRaises(ValueError, mape, x_2d, y_2d)
    #----------------- rmse() and mape() --------------------#


    #---------- generate_min_exponential_time() -------------#
    print("\n--- Testing generate_min_exponential_time(rates):")
    #-- Normal scenario
    np.random.seed(1717)
    rates = [0.2, 0.5, 0.7]
    t, idx = generate_min_exponential_time(rates)
    print("Observed minimum time {} at index {}".format(t, idx))
    t_expected, idx_expected = 0.3137434, 1
    assert np.allclose(t, t_expected)
    assert idx == idx_expected
    # Repeat the realization to check that the returned index is always 0 (because that's the only valid rate!)
    nevents_by_rate = np.zeros(len(rates))
    N = 10
    # Expected proportion of events for each exponential
    p = rates / np.nansum(rates)
    for i in range(N):
        t, idx = generate_min_exponential_time(rates)
        nevents_by_rate[idx] += 1
    # Observed proportions of events by server and type of event
    phat = 1.0 * nevents_by_rate / N
    se_phat = np.sqrt(p * (1 - p) / N)
    print("EXPECTED proportions of events by rate:\n{}".format(p))
    print("Observed proportions of events by rate on N={} generated events:\n{}".format(N, phat))
    print("Standard Errors on N={} generated events:\n{}".format(N, se_phat))
    assert np.allclose(phat, p, atol=3*se_phat)   # true probabilities should be contained in +/- 3 SE(phat) from phat

    #-- Case with some NaN and negative rate values
    np.random.seed(1717)
    t, idx = generate_min_exponential_time([0.2, np.nan, -0.7])
    t_expected, idx_expected = 2.1962040, 0
    print("Observed minimum time {} at index {}".format(t, idx))
    assert np.allclose(t, t_expected)
    assert idx == idx_expected
    # Repeat the realization to check that the returned index is always 0 (because that's the only valid rate!)
    for i in range(10):
        t, idx = generate_min_exponential_time([0.2, np.nan, -0.7])
        assert idx == idx_expected
    #---------- generate_min_exponential_time() -------------#



    #---------- compute_survival_probability() -------------#
    print("\n--- Testing compute_survival_probability(survival_times, columns=None, right_continuous=True):")

    # NOTE: The survival times must be sorted
    # Survival probability estimate with distinct survival times
    survival_times = [0.0, 1.3, 2.5, 5.0, 7.4]
    # P(T>t): right-continuous function
    proba_surv = compute_survival_probability(survival_times)
    expected_proba_surv = pd.DataFrame(np.c_[survival_times, [1.00, 0.75, 0.50, 0.25, 0.00]], columns=['t', 'P(T>t)'])
    assert all(proba_surv.columns == expected_proba_surv.columns)
    assert np.allclose(proba_surv, expected_proba_surv)

    # P(T>=t): left-continuous function
    proba_surv_left = compute_survival_probability(survival_times, right_continuous=False)
    expected_proba_surv_left = pd.DataFrame(np.c_[survival_times, [0.00] + expected_proba_surv['P(T>t)'][:-1].tolist()], columns=['t', 'P(T>=t)'])
    assert all(proba_surv_left.columns == expected_proba_surv_left.columns)
    assert np.allclose(proba_surv_left, expected_proba_surv_left)

    # P(T>t) with repeated observed survival times (typically occurring in discrete-time processes)
    # Note that each decrease of P(T>t) is still 1/N, even for the repeated time values.
    # And this is ok, because we just need to take the value of P(T>t) associated to the last occurrence of the repeated time value in order to get the estimate of the survival
    # probability function for that time value.
    survival_times = [0, 1, 1, 1, 2, 2, 3, 3, 7]
    proba_surv = compute_survival_probability(survival_times)
    expected_proba_surv = pd.DataFrame(np.c_[survival_times, [1.00, 0.875, 0.750, 0.625, 0.500, 0.375, 0.250, 0.125, 0.00]], columns=['t', 'P(T>t)'])
    assert np.allclose(proba_surv, expected_proba_surv)

    # Survival times are given unsorted
    # NOTE that NO error message is triggered (because checking for sorted values is computationally expensive)
    survival_times = [0.0, 2.5, 1.3, 7.4, 5.0]
    proba_surv = compute_survival_probability(survival_times)
    expected_proba_surv_wrong = pd.DataFrame(np.c_[survival_times, [1.00, 0.75, 0.50, 0.25, 0.00]], columns=['t', 'P(T>t)'])
    assert np.allclose(proba_surv, expected_proba_surv_wrong)

    # Check ValueErrors that should be raised by invalid function calls
    with pytest.raises(ValueError):
        # survival_times parameter is not a list
        compute_survival_probability(3)

        # survival_times has zero length
        compute_survival_probability([])

        # First element of survival times list is not 0.0
        compute_survival_probability([3.2, 5.0])
    # ---------- compute_survival_probability() -------------#


    #------------------- comb(n,k) -------------------------#
    print("\n--- Testing comb(n,k):")
    count = 0
    # Extreme cases
    assert comb(0,0) == 1; print(".", end=""); count += 1
    assert comb(1,0) == 1; print(".", end=""); count += 1
    assert comb(5,0) == 1; print(".", end=""); count += 1
    assert comb(5,5) == 1; print(".", end=""); count += 1
    assert comb(5,1) == 5; print(".", end=""); count += 1
    
    # Reference for the below technique of asserting that a ValueError is raised by the function call:
    # https://www.geeksforgeeks.org/python-assertion-error/
    try:
        comb(1,5)
    except ValueError:
        assert True; print(".", end=""); count += 1

    # Symmetry
    assert comb(6,3) == 20; print(".", end=""); count += 1
    assert comb(6,2) == 15; print(".", end=""); count += 1
    assert comb(6,4) == 15; print(".", end=""); count += 1
    
    # Very large numbers which using factorial give overflow
    assert comb(400,70) == 18929164090932404651386549010279950939591217550712274775589094772534627690086400; print(".", end=""); count += 1
    print("\nOK! {} tests passed.".format(count))
    #------------------- comb(n,k) -------------------------#


    #------------ all_combos_with_sum(R,C) -----------------#
    print("\n--- Testing all_combos_with_sum(R,C):")
    R = 3
    C = 20
    expected_count = comb(C+R-1,C)
    combos = all_combos_with_sum(R,C)
    count = 0
    while True:
        try:
            v = next(combos)
            assert len(v) == R, "v is of length R={} ({})".format(R, len(v))
            assert sum(v) == C, "the sum of elements of v is C={} ({})".format(C, sum(v))
            count += 1
            if count == 1:
                assert v == [20, 0, 0]
            if count == 2:
                assert v == [19, 1, 0]
            if count == 3:
                assert v == [19, 0, 1]
            if count == 4:
                assert v == [18, 2, 0]
            if count == 5:
                assert v == [18, 1, 1]
            if count == 6:
                assert v == [18, 0, 2]
            if count == expected_count:
                assert v == [0, 0, 20]
        except StopIteration:
            break
    combos.close()
    assert count == expected_count, "The number of combinations generated is {} ({})".format(expected_count, count)
    print("OK! {} combinations generated for R={}, C={}.".format(count, R, C))
    #------------ all_combos_with_sum(R,C) -----------------#


    #--------------- all_combos_with_max_limits(L) ------------------#
    print("\n--- Testing all_combos_with_max_limits(L):")
    L = (1, 2, 1)
    expected_count = np.prod([l + 1 for l in L])
    combos = all_combos_with_max_limits(L)
    count = 0
    while True:
        try:
            v = next(combos)
            assert len(v) == len(L), "v must be of length R={} ({})".format(len(L), len(v))
            assert [0 <= v[i] <= L[i] for i, _ in enumerate(L)], "each element of v must satisfy 0 <= v[i] <= L[i]: v={} L={}".format(v, L)
            count += 1
            if count == 1:
                assert v == [0, 0, 0]
            if count == 2:
                assert v == [0, 0, 1]
            if count == 3:
                assert v == [0, 1, 0]
            if count == 4:
                assert v == [0, 1, 1]
            if count == 5:
                assert v == [0, 2, 0]
            if count == 6:
                assert v == [0, 2, 1]
            if count == 7:
                assert v == [1, 0, 0]
            if count == 8:
                assert v == [1, 0, 1]
            if count == 9:
                assert v == [1, 1, 0]
            if count == 10:
                assert v == [1, 1, 1]
            if count == 11:
                assert v == [1, 2, 0]
            if count == 12:
                assert v == [1, 2, 1]
        except StopIteration:
            break
    combos.close()
    assert count == expected_count, "The number of combinations generated must be {} ({})".format(expected_count, count)
    print("OK! {} combinations generated for L={}.".format(count, L))

    # Case with ONE dimension
    L = [4]
    expected_count = L[0] + 1
    combos = all_combos_with_max_limits(L)
    count = 0
    values = []
    while True:
        try:
            v = next(combos)
            assert len(v) == len(L), "v must be of length R={} ({})".format(len(L), len(v))
            assert [0 <= v[i] <= L[i] for i, _ in enumerate(L)], "each element of v must satisfy 0 <= v[i] <= L[i]: v={} L={}".format(v, L)
            values += v
            count += 1
        except StopIteration:
            break
    combos.close()
    assert count == expected_count, "The number of combinations generated must be {} ({})".format(expected_count, count)
    assert values == list(range(L[0] + 1))
    print("OK! {} combinations generated for L={}.".format(count, L))
    #--------------- all_combos_with_max_limits(L) ------------------#


    #------------------ compute_set_of_frequent_states_with_zero_reward() ------------------#
    print("\n--- Testing compute_set_of_frequent_states_with_zero_reward(states, rewards, threshold=0.05):")

    print("Case where:\n(i) Not all the states in the system are visited.\n(ii) One of the state counts is exactly equal to the threshold value, so that we test that the threshold condition should be strictly overcome in order to be included in set A.")
    # Number of states in the system
    nS = 6
    np.random.seed(13)
    states = [0]*20 + [3]*18 + [1]*10 + [2]*3 + [5]*13
    rewards = [0]*20 + [1]*4 + [0]*14 + [0]*(10 + 3 + 13)
    nsteps = len(states)
    assert len(states) == len(rewards)
    order = np.random.permutation(nsteps)
    states = [states[o] for o in order]
    rewards = [rewards[o] for o in order]
    set_A = compute_set_of_frequent_states_with_zero_reward(states, rewards)
    print(f"States and rewards observed in the trajectory (n={len(states)}):\n{np.c_[states, rewards].T}")
    print(f"Distribution of states:\n{pd.Series(states).value_counts(sort=False)}")
    print(f"Distribution of states with zero reward:\n{pd.Series(states)[pd.Series(rewards)==0].value_counts(sort=False)}")
    print(f"Default threshold (5% of {sum(pd.Series(rewards)==0)}) = {0.05*sum(pd.Series(rewards)==0)}")
    print(f"Absorption set A: {set_A}")
    assert set_A == {0, 1, 3, 5}, f"The absorption set A must be equal to {{0, 1, 3, 4}}: {set_A}"
    #------------------ compute_set_of_frequent_states_with_zero_reward() ------------------#


    #---------------- compute_state_value_function_from_transition_matrix() ----------------#
    # Tests on a 1D gridworld under different learning tasks (EPISODIC or CONTINUING) and different learning criteria (DISCOUNTED or AVERAGE)
    print("\n--- Testing compute_state_value_function_from_transition_matrix(P, b, g, gamma) for EPISODIC and CONTINUING learning tasks:")

    #-- DATA PREPARATION
    from Python.lib.environments.gridworlds import EnvGridworld1D
    from Python.lib.agents.policies import probabilistic

    # Environment with start state at 0
    nS = 5
    isd = np.zeros(nS); isd[0] = 1.0
    env1d = EnvGridworld1D(length=nS, rewards_dict={nS-1: +1.0}, reward_default=0.0, initial_state_distribution=isd)
    print(f"Policy in {nS}-state gridworld:")
    for k, v in env1d.P.items():
        print(f"State {k}: {v}")
    print(f"Terminal states in {nS}-state gridworld: {env1d.getTerminalStates()}")
    print(f"Rewards: {env1d.getRewardsDict()}")
    print(f"Start state distribution: {env1d.isd}")

    # Policy for all states except state 0 where with probability 1 it goes to state 1 (i.e. it goes right)
    policy_probabilities = [0.1, 0.9]
    policy = probabilistic.PolGenericDiscrete(env1d, dict({0: [0.0, 1.0]}), policy_default=policy_probabilities)
    P_epi, P_con, b_epi, b_con, g, mu = compute_transition_matrices(env1d, policy)
    print(f"EPISODIC transition probability matrix, P_epi:\n{P_epi}")
    print(f"CONTINUING transition probability matrix, P_con:\n{P_con}")
    print(f"EPISODIC expected one-step reward vector, b_epi:\n{b_epi}")
    print(f"CONTINUING expected one-step reward vector, b_con:\n{b_con}")
    print(f"CONTINUING expected average reward, g:\n{g}")
    print(f"Stationary probability distribution for the CONTINUING learning task in the 1D gridworld: {mu}")

    # Discount factor for the EPISODIC learning task
    gamma = 0.9
    #-- DATA PREPARATION

    #-- TESTS
    # EPISODIC learning task under the DISCOUNTED reward criterion
    V = compute_state_value_function_from_transition_matrix(P_epi, b_epi, gamma=gamma)
    print(f"\nState value function for EPISODIC learning task under the DISCOUNTED reward criterion with gamma = {gamma}: {V}")
    assert np.allclose(V, np.array([6.82117389,  7.5790821 ,  8.59898327,  9.77390849, 10.]))

    # CONTINUING learning task under the DISCOUNTED reward criterion
    V = compute_state_value_function_from_transition_matrix(P_con, b_con, bias=g, gamma=gamma)
    print(f"State value function for CONTINUING learning task under the DISCOUNTED reward criterion with gamma = {gamma}: {V}")
    assert np.allclose(V, np.array([-0.22428486, -0.05491426,  0.17300442,  0.43556688, -0.37671823]))

    # CONTINUING learning task under the AVERAGE reward criterion
    V = compute_state_value_function_from_transition_matrix(P_con, b_con, bias=g)
    print(f"State value function for CONTINUING learning task under the AVERAGE reward criterion with gamma = {gamma}: {V}")
    assert np.allclose(V, np.array([-0.19904054, -0.02417846,  0.18954186,  0.40757976, -0.37390261]))
    #-- TESTS
    #---------------- compute_state_value_function_from_transition_matrix() ----------------#


    #------------ stationary_distribution_product_form --------------#
    import matplotlib.pyplot as plt
    print("\n--- Testing stationary_distribution_product_form(capacity, rhos, func_prod_birthdeath):")
    print("IMPORTANT: This test does NOT test the correctness of the theoretic distribution, it is just useful for regression tests")
    R = 3
    C = 3
    n_expected = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2],
                [3, 0, 0],
                [2, 1, 0],
                [2, 0, 1],
                [1, 2, 0],
                [1, 1, 1],
                [1, 0, 2],
                [0, 3, 0],
                [0, 2, 1],
                [0, 1, 2],
                [0, 0, 3]
                ]

    print("Test #1: all process intensities rhos are equal (to 0.5)")
    rhos_equal = [0.5, 0.5, 0.5]
    dist_expected = [
                    0.190476,
                    0.095238,
                    0.095238,
                    0.095238,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810
                    ]           
    n, dist = stationary_distribution_product_form(C, rhos_equal, func_prod_birthdeath)
    print("State space for R={}, C={}: {}".format(R, C, len(dist)))
    print("Distribution for rhos={} computed by the tested function:".format(rhos_equal))
    [print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(n, dist))]
    print("---------------")
    print("Sum (should be 1): p={:.6f}".format(sum(dist)))
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function is 1 ({.6f})".format(sum(dist))
    assert all([x == x_expected for x, x_expected in zip(n, n_expected)]), "The expected event space is verified"
    assert np.allclose(dist, dist_expected, atol=1E-6), "The expected distribution is verified"

    # Multinomial distribution on these events
    # Notes:
    # - sample_size is the number of experiments to run for the multinomial distribution
    # in our case, it's our sample size.
    # - the resulting array gives the number of times we get the index value out of all possible indices
    # from 0 ... length(dist)-1, so we can use it as the number of times to use the n array associated
    # to each index as starting point of the simulation
    sample_size = 1000
    print("Generating {} samples with the given Multinomial distribution".format(sample_size))
    sample = np.random.multinomial(sample_size, dist, size=1)
    print("No. of times each array combination n appears in the sample:\n{}".format(sample[0]))
    plt.figure()
    plt.plot(dist, 'r.-')
    plt.plot(sample[0]/sample_size, 'b.')
    plt.legend(["Theoretic distribution", "Observed frequency of values sampled from the theoretic distribution"])
    plt.title("Theoretical vs. Observed product-form stationary distribution for a multi-server queue system\nR={}, C={}, rhos={} (ALL EQUAL) (sample size = {})".format(R, C, rhos_equal, sample_size))

    print("Test #2: process intensities rhos are different (smaller than 1)")
    rhos_diff = [0.8, 0.7, 0.4]
    dist_expected = [
                    0.124611,
                    0.099688,
                    0.087227,
                    0.049844,
                    0.079751,
                    0.069782,
                    0.039875,
                    0.061059,
                    0.034891,
                    0.019938,
                    0.063801,
                    0.055826,
                    0.031900,
                    0.048847,
                    0.027913,
                    0.015950,
                    0.042741,
                    0.024424,
                    0.013956,
                    0.007975
                    ]
    n, dist = stationary_distribution_product_form(C, rhos_diff, func_prod_birthdeath)
    print("State space for R={}, C={}: {}".format(R, C, len(dist)))
    print("Distribution for rhos={} computed by the tested function:".format(rhos_diff))
    [print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(n, dist))]
    print("---------------")
    print("Sum (should be 1): p={:.6f}".format(sum(dist)))
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function is 1 ({.6f})".format(sum(dist))
    assert all([x == x_expected for x, x_expected in zip(n, n_expected)]), "The expected event space is verified"
    assert np.allclose(dist, dist_expected, atol=1E-6), "The expected distribution is verified"

    # Multinomial distribution on these events
    # Notes:
    # - sample_size is the number of experiments to run for the multinomial distribution
    # in our case, it's our sample size.
    # - the resulting array gives the number of times we get the index value out of all possible indices
    # from 0 ... length(dist)-1, so we can use it as the number of times to use the n array associated
    # to each index as starting point of the simulation
    sample_size = 1000
    print("Generating {} samples with the given distribution".format(sample_size))
    sample = np.random.multinomial(sample_size, dist, size=1)
    print("No. of times each array combination n appears in the sample:\n{}".format(sample[0]))
    plt.figure()
    plt.plot(dist, 'r.-')
    plt.plot(sample[0]/sample_size, 'b.')
    plt.legend(["Theoretic distribution", "Observed frequency of values sampled from the theoretic distribution"])
    plt.title("Theoretical vs. Observed product-form stationary distribution for a multi-server queue system\nR={}, C={}, rhos={} (ALL DIFFERENT) (sample size = {})".format(R, C, rhos_equal, sample_size))
    #------------ stationary_distribution_product_form --------------#


    #------ compute_blocking_probability_birth_death_process --------#
    print("\n--- Testing compute_blocking_probability_birth_death_process(rhos, capacity):")
    print("Single server system")
    print("Test #1: rho = 0.7")
    rhos = [0.7]
    C = 3
    p_expected = 0.135413
    p = compute_blocking_probability_birth_death_process(rhos, C)
    print("---------------")
    print("Estimated p = {:.6f}".format(p))
    print("True p = {:.6f}".format(p_expected))
    assert np.isclose(p, p_expected), "The expected blocking probability is verified"

    print("\nMulti-server systems:")
    C = 3

    print("Test #1: all process intensities rhos are equal (to 0.5)")
    rhos_equal = [0.5, 0.5, 0.5]
    p_expected = 0.238095
    p = compute_blocking_probability_birth_death_process(rhos_equal, C)
    print("---------------")
    print("Estimated p = {:.6f}".format(p))
    print("True p = {:.6f}".format(p_expected))
    assert np.isclose(p, p_expected), "The expected blocking probability is verified"

    print("Test #2: process intensities rhos are different (smaller than 1)")
    rhos_diff = [0.8, 0.7, 0.4]
    p_expected = 0.333333
    p = compute_blocking_probability_birth_death_process(rhos_diff, C)
    print("---------------")
    print("Estimated p = {:.6f}".format(p))
    print("True p = {:.6f}".format(p_expected))
    assert np.isclose(p, p_expected), "The expected blocking probability is verified"
    #----------------- compute_blocking_probability -----------------#
