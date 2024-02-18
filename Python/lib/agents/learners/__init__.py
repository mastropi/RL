# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:37:08 2021

@author: Daniel Mastropietro
@description: Definition of classes that are common to all generic learners on discrete-time discrete state and actions,
where 'generic' means that:
- they could either be part of a continuing or episodic task.
- the number of states and actions of the environment where they learn may not be fixed (e.g. capacity of a queue).
- the times observed during the trajectory experienced by the agent interacting with the environment could be either
continuous or discrete (these are normally stored in attribute self.times).
HOWEVER, the learning time is ALWAYS discrete (normally stored in attribute self._t).

Classes inheriting from this class should define the following methods:
- reset_value_functions()
"""

import copy
from enum import Enum, unique

import numpy as np

MIN_COUNT = 1  # Minimum state count to start shrinking alpha


@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class ResetMethod(Enum):
    ALLZEROS = 1
    RANDOM_UNIFORM = 2
    RANDOM_NORMAL = 3

@unique
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2

@unique
class LearnerTypes(Enum):
    "Types of learners that can be defined in learner classes"
    V = 'state_value'
    Q = 'state_action_value'
    P = 'policy'

@unique
class LearningCriterion(Enum):
    """
    Whether the learning criterion is the discounted return, the average reward, or the total reward

    When learning is by the average criterion, the value functions learned are the *bias* or *difference* values.
    Ref: Sutton (2018), pag. 250

    Values are sorted by what is deemed "most to least common scenario".
    """
    DISCOUNTED = 1
    AVERAGE = 2
    TOTAL = 3

@unique
class LearningTask(Enum):
    """
    Whether the learning task is episodic (i.e. learning happens in episodes of finite length T) or
    continuing (i.e. learning happens continuously until a maximum number of steps is observed, or until
    the estimation of the value functions has reached an accepted error).

    Values are sorted by what is deemed "most to least common scenario".
    """
    EPISODIC = 1
    CONTINUING = 2

# Identity function: used to define the default transformation function of the counter (n) that adjusts alpha
identity = lambda x: x


class GenericLearner:
    """
    Class defining methods that are generic to ALL learners.

    IMPORTANT: Before using any learner the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: once when the learner is constructed
    and once prior to the first simulation.
    """

    def __init__(self, env,
                 criterion=LearningCriterion.DISCOUNTED,
                 task=LearningTask.EPISODIC,
                 alpha: float=1.0,
                 adjust_alpha=False,
                 func_adjust_alpha=None,
                 min_count_to_update_alpha=0, min_time_to_update_alpha=0,
                 alpha_min=0.):
        """
        Parameters:
        env: gym.Env
            Environment where learning takes place.
            The environment needs not be "discrete" in the sense the gym package uses discrete, namely that there is
            a pre-defined number of states (as is the case in the EnvironmentDiscrete environment of gym).

        criterion: (opt) LearningCriterion
            The criterion used to learn the value functions, either DISCOUNTED (for episodic tasks with discount factor gamma < 1)
            or AVERAGE, for the average reward criterion (for continuing tasks, with discount factor gamma = 1).
            default: LearningCriterion.DISCOUNTED

        task: (opt) LearningTask
            Type of learning task as defined in the LearningTask enum class.
            Typical alternatives are "episodic learning task" and "continuing learning task".
            default: LearningTask.EPISODIC

        alpha: (opt) positive float
            Initial learning rate.
            default: 1.0

        adjust_alpha: (opt) bool
            Whether alpha should be updated when the methods that are responsible for updating alpha are called.
            default: False

        func_adjust_alpha: (opt) callable
            Function that is used on the counter of whatever KPI is used to divide the initial alpha
            when the methods responsible for updating alpha perform the update operation.
            Ex: `np.sqrt`, in which case alpha is updated as alpha_start / np.sqrt(n) where n is the counter of the KPI
            default: None, in which case the identity function is applied to n when dividing alpha

        min_count_to_update_alpha: (opt) int
            Minimum count of a state-action pair at which alpha starts to be updated by the update_learning_rate_by_state_action_count(s,a) method.
            default: 0

        min_time_to_update_alpha: (opt) int
            Minimum learning time step at which alpha starts to be updated by the update_learning_rate_by_learning_epoch() method.
            default: 0
        """
        self.env = env
        self.criterion = criterion
        self.task = task
        self.alpha = alpha is None and 1.0 or alpha          # Initial and maximum learning rate
        self.adjust_alpha = adjust_alpha is None and False or adjust_alpha

        self.func_adjust_alpha = func_adjust_alpha is None and identity or func_adjust_alpha
        self.min_count_to_update_alpha = min_count_to_update_alpha is None and 0 or min_count_to_update_alpha
        self.min_time_to_update_alpha = min_time_to_update_alpha is None and 0 or min_time_to_update_alpha
        self.alpha_min = alpha_min is None and 0.0 or alpha_min # Used when adjust_alpha=True

        # Observed state, action, and reward at the latest learning time
        self.state = None           # S(t): state BEFORE taking the action A(t) and receiving reward R(t+1)
        self.action = None          # A(t): action taken at state S(t)
        self.reward = None          # R(t+1): reward received by the agent when taking action A(t) on state S(t)

        # Average reward for contiuing tasks
        # Only used when criterion = LearningCriterion.AVERAGE
        self.average_reward = 0.0   # Average reward observed so far: useful for continuing tasks where the average reward is used to compute the differential value functions

        # Trajectory history: information of the times, states, actions, and rewards stored as part of the learning process
        # (so that we can be retrieve it for e.g. analysis or plotting)
        self.times = []         # This time is either a DISCRETE time or a CONTINUOUS time depending on what
                                # type of process (discrete or continuous) is used to model the evolution of the environment
                                # with which the learner interacts.
                                # The time is associated to the state, action and reward stored in the respective attributes.
        self.states = []        # Stores S(t), the state at each time t stored in `times`
        self.actions = []       # Stores A(t), the action taken at state S(t)
        self.rewards = []       # Stores R(t+1) = R(S(t), A(t)), the reward received after taking action A(t) on state S(t)
                                # Note that we initialize the rewards list as EMPTY, exactly as we do with the states and actions lists,
                                # because we leave the responsibility to the ACTUAL learner to fill in the first reward as 0
                                # (if needed) if the learner wishes to reference to the state-action-reward sequence
                                # by using the indices associated to the usual sequence used to represent it, namely:
                                # S(0), A(0), R(1), S(1), A(1), R(2), etc.
                                # However, this notation is NOT really necessary, the learner may well refer to this sequence as:
                                # S(0), A(0), R(0), S(1), A(1), R(1), etc.
                                # i.e. using the same indices in S, A and R to refer to each state-action-reward sequence.
                                # In fact, the main reason we initialize rewards as empty is the currently implemented
                                # management of the rewards over ALL episodes in episodic learning tasks, which is done
                                # typically (and mainly) in the store_trajectory_at_episode_end() method in the Learner class,
                                # where self.rewards is updated by concatenating self._rewards, i.e. the list of rewards observed
                                # during the latest episode. In fact, if we initialize self.rewards here as [0], such concatenation
                                # will create problems down the line, because e.g. the first concatenation will result in a weird
                                # list as e.g. [0, [0, 0, 0, 1]], which would yield the error message "cannot concatenate zero-length arrays"
                                # or similar message when calling np.concatenate(self.reward) when e.g. computing the average reward
                                # over all rewards stored in the self.reward list.
                                # For more information, see also the comment written above the initialization of the self._rewards list
                                # containing the episodic rewards in the Learner class.
                                # Taking a look at the Learner.store_trajectory_at_episode_end() method may also be helpful
                                # to better understand the issues just described.

        # Count of visited states and visited state-actions
        # Depending on the type of learner (e.g. learner of V or learner of Q) one ore the other will be updated.
        self.dict_state_counts = dict()
        self.dict_state_action_counts = dict()

        # Learning epoch and learning rate at that epoch
        self._t = 0              # Int: this value is expected to be updated by the user, whenever learning takes place
        self._alpha = self.alpha # Float: current learning rate (at the learning time `self._t`), REGARDLESS of the state and action the process has just visited.

        # Information about the historical learning rates
        self.alpha_mean = []    # Average alpha. This is currently (2024/01/06) NOT used.
        self.alphas = []        # List of alphas used during the learning process, which is updated by EXPLICITLY calling store_learning_rate()
                                # In principle, this list can contain anything, either scalars or e.g. lists (which are useful when storing the alpha used by state))
                                # When a scalar is stored, the scalar value would normally correspond to the alpha value used when learning at the moment when
                                # the store_learning_rate() method was called, REGARDLESS of the state and action the process just visited.

    def reset(self, reset_learning_epoch=True, reset_alphas=True, reset_value_functions=True, reset_trajectory=True, reset_counts=True):
        """
        Resets the variables that store information about the learning process

        Parameters:
        reset_learning_epoch: (opt) bool
            Whether to reset the learning time counter.
            default: True

        reset_alphas: (opt) bool
            Whether to reset the learning rate and its history.
            default: True

        reset_value_functions: (opt) bool
            Whether to reset all the value functions to their initial estimates.
            default: True

        reset_trajectory: (opt) bool
            Whether to reset the information of the trajectory (states, actions, rewards).
            default: True

        reset_counts: (opt) bool
            Whether to reset the counters (of e.g. the states or the action-states).
            default: True
        """
        # Reset the latest learning state, action, and reward
        self.state = None
        self.action = None
        self.reward = None

        # Reset the return G
        self.reset_return()

        if reset_learning_epoch:
            self._t = 0

        if reset_alphas:
            self._alpha = self.alpha
            self.alpha_mean = []
            self.alphas = []

        if reset_value_functions:
            # Only reset the initial estimates of the value functions when requested
            self.reset_value_functions()
            # Reset the average reward
            self.average_reward = 0.0

        if reset_trajectory:
            self.reset_trajectory()

        if reset_counts:
            self.reset_counts()

    def reset_trajectory(self):
        self.times = []
        self.states = []
        self.actions = []
        self.rewards = []

    # Overrides superclass method
    def reset_counts(self):
        "Resets the dictionaries containing the number of times a state and a state-action is visited"
        self.dict_state_counts = dict()
        self.dict_state_action_counts = dict()

    def reset_return(self):
        "Resets the observed return. Method to be implemented by subclasses if the return G is defined by the subclass."
        pass

    def reset_value_functions(self):
        # Method to be implemented by subclasses
        raise NotImplementedError

    def store_learning_rate(self, alpha=None):
        """
        Stores as part of the alpha history (self.alphas) the current learning rate alpha (self._alpha) that has been used for learning
        when this method is called, UNLESS parameter `alpha` is given in which case that alpha value (which can be a complex object, such a list or array)
        is stored.

        Note that the values of the self.alphas list may be indexed by many different things...
        For instance, if learning takes place at every visited state-action, then there will probably be an alpha
        for each visited state-action. But if learning happens at the end of an epoch or episode, the stored alphas
        will probably be indexed by epoch number.
        It is however difficult to store the index together with the alpha, because the index structure may change
        from one situation to the other, as just described.

        Arguments:
        alpha: object or float
            The alpha "value" to store which can be e.g. a list of alpha values (e.g. one alpha value per state or per state-action of the environment).
            This latter case is useful if we want to keep track of how the alpha values changed by state or state-action as the learning process progressed.
            default: None, in which case the (float) value of the self._alpha attribute is stored.
        """
        if alpha is None:
            self.alphas += [self._alpha]
        else:
            # Add a DEEP copy of alpha in case it is an object, o.w. all elements in self.alphas will end up to be the same!!
            # Note that copy.deecopy(<scalar>) works fine
            self.alphas += [copy.deepcopy(alpha)]

    def update_learning_epoch(self):
        """
        Increments the count of learning epoch by one

        This method is expected to be called by the user whenever they want to record that learning took place.
        It can be used to update the learning rate alpha by the number of learning epochs
        (see update_learning_rate_by_learning_epoch()).
        """
        self._t += 1

    def update_learning_rate_by_state_action_count(self, state, action):
        """
        Updates the learning rate at the current learning time, given the visit count of the given state and action

        The update is done ONLY when the attribute self.adjust_alpha is True.

        Arguments:
        state: Environment dependent
            State visited by the environment used to retrieve the visited count to update alpha.

        action: Environment dependent
            Action received by the environment used to retrieve the visited count to update alpha.

        Return: float
        The updated value of alpha based on the visited count of the state and action.
        """
        #with np.printoptions(precision=4):
        #    print("Before updating alpha: state {}: state_action_count={:.0f}, alpha>={}: alpha={}\n{}" \
        #          .format(state, self.dict_state_action_count[str(state)], self.alpha_min, self.alphas[str(state)]))
        if self.adjust_alpha:
            state_action_count = self.getStateActionCount(state, action)
            time_divisor = self.func_adjust_alpha( max(1, state_action_count - self.min_count_to_update_alpha) )
                ## Note: We don't sum anything to the difference count - min_count because we want to have the same
                ## time_divisor value when min_count = 0, in which case the first update occurs when count = 2
                ## (yielding time_divisor = 2), as when min_count > 0 (e.g. if min_count = 5, then we would have the
                ## first update at count = 7, with time_divisor = 2 as well (= 7 - 5)).
            print("\tUpdating alpha... state={}, action={}: time divisor = {}".format(state, action, time_divisor))
            alpha_prev = self._alpha
            self._alpha = max(self.alpha_min, self.alpha / time_divisor)
            print("\t\tstate {}: alpha: {} -> {}".format(state, alpha_prev, self._alpha))
        else:
            self._alpha = self.alpha

        return self._alpha

    def update_learning_rate_by_learning_epoch(self):
        """
        Updates the learning rate at the current learning epoch by the number of times learning took place already,
        independently of the visit count of each state and action.

        The update is done ONLY when the attribute self.adjust_alpha is True.

        Return: float
        The updated value of alpha.
        """
        if self.adjust_alpha:
            time_divisor = self.func_adjust_alpha( max(1, self._t - self.min_time_to_update_alpha) )
                ## See comment in method update_learning_rate_by_state_action_count() about not adding any constant to the difference time - min_time
            print("\tUpdating alpha by learning epoch: time divisor = {}".format(time_divisor))
            alpha_prev = self._alpha
            self._alpha = max( self.alpha_min, self.alpha / time_divisor )
            print("\t\talpha: {} -> {}".format(alpha_prev, self._alpha))
        else:
            self._alpha = self.alpha

        return self._alpha

    def update_trajectory(self, t, state, action, reward):
        """
        Records the state S(t) on which an action is taken, the taken action A(t), and the observed reward R(t)
        for going to state S(t+1), and updates the history of this trajectory.

        Arguments:
        t: int
            Time at which the given, state, action, and reward occur.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self._update_trajectory_history(t)
        # NOTE: # We need to update the average reward AFTER updating the trajectory because the average reward update
        # uses the length of the historic rewards stored (as the number of sampled rewards on which the average is computed).
        self.update_average_reward()

    def _update_trajectory_history(self, time):
        """
        Updates the history of the visited states, actions taken, and observed rewards at the given simulation time

        Arguments:
        time: int or float
            Discrete or continuous time to which the stored state, action, and reward are associated.
        """
        self.times += [time]
        self.states += [self.state]
        self.actions += [self.action]
        self.rewards += [self.reward]

    def update_state_action_counts(self, state, action):
        "Updates the number of times the given state and action have been visited during the learning process"
        # Create a string version of the state, so that we can store lists as dictionary keys
        # which are actually not accepted as dictionary keys (with the error message "list type is unhashable").
        # e.g. [3, 1, 5] becomes '[3, 1, 5]'
        state_action_str = str( (state, action) )
        if state_action_str in self.dict_state_action_counts.keys():
            # Update the value of the dictionary key representing the state-action that is visited now
            self.dict_state_action_counts[state_action_str] += 1
        else:
            # Add the state-action as a new dictionary key since it is the first time it is visited
            self.dict_state_action_counts[state_action_str] = 1

    def update_average_reward(self):
        # TODO: (2023/08/31) According to the algorithm for learning the average reward presented in Sutton (2018), pag. 251, a better learner of the average reward uses a separate learning rate which is applied on the delta error... Implement this.
        # I think this should probably be implemented within each learner inheriting from this class...(?) because they store the delta error value to use.
        n_rewards_observed_so_far = len(self.rewards) - 1   # We subtract 1 to the length of self.rewards because the first element in self.rewards is a fictitious reward of 0 (see its initialization in the constructor)
        self.average_reward += (self.reward - self.average_reward) / max(1, n_rewards_observed_so_far)  # `max(1, ...)` to avoid division by 0 if we are storing the very first reward

    def getAverageLearningRates(self):
        return self.alpha_mean

    def getLearningRates(self):
        return self.alphas

    def getLearningRate(self):
        return self._alpha

    def getInitialLearningRate(self):
        return self.alpha

    def getLearningCriterion(self):
        "Returns the learning criterion, i.e. one of the possible values of LearningCriterion enum (e.g. AVERAGE, DISCOUNTED)"
        return self.criterion

    def getLearningTask(self):
        "Returns the learning task type, i.e. one of the possible values of LearningTask enum (e.g. CONTINUING, EPISODIC)"
        return self.task

    def getLearningEpoch(self):
        "Returns the number of epochs at which learning took place according to the learning epoch attribute"
        return self._t

    def getStateActionCount(self, state, action):
        "Returns the number of times the given state and action has been visited during the learning process"
        key = str( (state, action) )
        return self.dict_state_action_counts.get(key, 0)

    def getTimes(self):
        return self.times

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getRewards(self):
        return self.rewards

    def getAverageReward(self):
        return self.average_reward

    def setInitialLearningRate(self, alpha):
        "Sets the initial learning rate"
        self.alpha = alpha

    def setAverageReward(self, average_reward):
        self.average_reward = average_reward
