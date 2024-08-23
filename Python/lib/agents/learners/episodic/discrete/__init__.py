# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:15:04 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all episodic discrete-time learners on discrete FINITELY many
and FIXED states and actions. This includes learning on environments defined typically in the gym.envs.toy_text.discrete
module. On the contrary, any environment whose states may vary along learning of a policy (e.g. queues with variable
capacity) are excluded from this set of environments where learners inheriting from this class can learn on.

All learner classes inheriting from this class:
a) SHOULD implement the following attributes:
    - V --> an object containing the information on how the state value function is ESTIMATED.
    - Q --> an object containing the information on how the state-action value function is ESTIMATED.
    - A --> an object containing the information on how the advantage function is ESTIMATED.
    - gamma --> the reward discount parameter
b) COULD implement the following methods:
    - reset() --> resets the state of the learner to start learning anew
        (e.g. all the estimates of the value functions are reset to 0)
    - setParams() --> set the parameters of the learner (if any)
    - learn() --> prediction problem: learns the state value function under the current policy (V(s))
    - learn_pred_Q() --> prediction problem: learns the action-value function under the current policy (Q(s,a))
    - learn_ctrl_policy() --> control problem: learns the optimal policy
    - getStateCounts(first_visit) --> returns the state counts over all run episodes, optionally the first visit counts.
    - getV() --> returns the state value function
    - getQ() --> returns the state-action value function
    - getA() --> returns the advantage function (for each state and action)
"""

import warnings
from enum import Enum, unique
from collections import deque

import numpy as np

from Python.lib.environments import EnvironmentDiscrete
from Python.lib.agents.learners import GenericLearner, LearningCriterion, LearningTask, ResetMethod

MIN_EPISODE = 1  # Minimum episode count to start shrinking alpha
MAX_EPISODE_FOR_ALPHA_MIN = None  # Maximum episode on which the minimum alpha value is applied, before continuing to decrease alpha again further (from the alpha value observed at the MAX_EPISODE_FOR_ALPHA_MIN episode)


@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2


class Learner(GenericLearner):
    """
    Class defining methods that are generic to ALL episodic discrete-time learners.

    IMPORTANT: Before using any learner, the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: once when the learner is constructed
    and once prior to the first simulation, making e.g. the episode method be equal to 2 at
    the first simulation as opposed to the correct value 1.

    HOW TO RETRIEVE THE TRAJECTORY INFORMATION:
    It is suggested to retrieve the trajectory information as follows:
        [(t, r, s, a) for t, r, s, a in zip(learner.getTimes(), learner.getRewards(), learner.getStates(), learner.getActions()]
    as this is how the states, actions and rewards are aligned by this Learner class, i.e. each entry in the list returned by
    the getter methods just used (getTimes, getRewards, etc.) correspond to the same time.
    In particular R(t) (stored in self.rewards[t]) is the reward the agent received WHEN visiting state S(t) (stored in self.states[t])
    and A(t) (stored in self.actions[t]) is the action taken by the action AFTER visiting state S(t) (stored in self.states[t]).
    So the above order corresponds to showing the trajectory as the sequence:
        R(0), S(0), A(0), R(1), S(1), A(1), R(2), ...
    making R(t) be the reward received when visiting S(t).
    """
    def __init__(self, env,
                 criterion: LearningCriterion=LearningCriterion.DISCOUNTED,
                 task: LearningTask=LearningTask.EPISODIC,
                 alpha: float=1.0,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT, adjust_alpha_by_episode=False,
                 func_adjust_alpha=None,
                 alpha_min=0.,
                 min_count_to_update_alpha=1, min_time_to_update_alpha=0,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False):
        """
        Parameters:
        env: EnvironmentDiscrete
            Environment where learning takes place.
            It should have the following methods defined:
            - getNumStates() which returns the number of fixed states in the environment.
            - getState() which returns the current state of the environment.
            - getReward() which returns the reward received by the agent when visiting a state in the environment.

        criterion: (opt) LearningCriterion
            See documentation of the super class.
            default: LearningCriterion.DISCOUNTED

        task: (opt) LearningTask
            See documentation of the super class.
            default: LearningTask.EPISODIC

        alpha: (opt) positive float
            See documentation of the super class.
            default: 1.0

        adjust_alpha: (opt) bool
            See documentation of the super class.
            default: False

        learner_type: (opt) LearnerType *** NOT YET IMPLEMENTED BUT COULD BE A GOOD IDEA TO AVOID DEFINING THE AlphaUpdateType...? ***
            Type of learner. E.g. LearnerType.TD, LearnerType.MC.

        alpha_update_type: (opt) AlphaUpdateType
            How alpha is updated, e.g. AlphaUpdateType.FIRST_STATE_VISIT, AlphaUpdateType.EVERY_STATE_VISIT
            This value defines the denominator when updating alpha for each state as alpha/n, where alpha
            is the initial learning rate (passed as parameter alpha) and n is the number of FIRST or EVERY visit
            to the state, depending on the value of the alpha_update_type parameter.

        adjust_alpha_by_episode: (opt) bool
            Whether the learning rate alpha is adjusted by number of episodes run, as opposed to by number of visits.
            default: False

        func_adjust_alpha: (opt) callable
            See documentation of the super class.
            default: None

        alpha_min: (opt) float
            See documentation of the super class.
            default: False

        min_count_to_update_alpha: (opt) int
            Minimum count of a state or state-action pair at which alpha starts to be adjusted (decreased).

        min_time_to_update_alpha: (opt) int
            Minimum learning time step at which alpha starts to be updated by the update_learning_rate_by_learning_epoch() method.

        reset_method: (opt) ResetMethod
            Method to use to reset the value function at the beginning of the experiment.
            default: ResetMethod.ALLZEROS

        reset_params: (opt) dict
            Dictionary defining the parameters to use by the pseudo-random number generator to reset the value function
            at the beginning of the experiment.
            default: None

        reset_seed: (opt) int
            Seed to use for the random reinitialization of the value function (if reset_method so specifies it).
            default: None

        store_history_over_all_episodes: (opt) bool
            Whether to store in the object (in the attributes of the super class) the history over ALL episodes
            or just the history of the latest observed episode.
            The former is useful when we need to do computations on the whole observed history, possibly
            for learning under the average reward criterion (although its validity is arguable, as rewards
            observed in an episode may not be related to rewards observed in future episodes, thus making
            its combination invalid).
            Note also that if storing the whole history is requested, the occupied memory will be large
            if a large number of episodes are run...
            default: False
        """
        #        if not isinstance(env, EnvironmentDiscrete):
        #            raise TypeError("The environment must be of type {} from the {} module ({})" \
        #                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        #        if not isinstance(alpha_update_type, AlphaUpdateType):
        #            raise TypeError("The alpha_update_type of learner must be of type {} from the {} module ({})" \
        #                            .format(AlphaUpdateType.__name__, AlphaUpdateType.__module__, alpha_update_type.__class__))
        super().__init__(env, criterion=criterion, task=task, alpha=alpha, adjust_alpha=adjust_alpha, func_adjust_alpha=func_adjust_alpha, alpha_min=alpha_min,
                         min_count_to_update_alpha=min_count_to_update_alpha, min_time_to_update_alpha=min_time_to_update_alpha)
        self.alpha_update_type = alpha_update_type
        self.adjust_alpha_by_episode = adjust_alpha_by_episode

        # Episode counter
        self.episode = 0

        # Current learning rate across states
        self._alphas = self.alpha * np.ones(self.env.getNumStates())
        # Learning rate at episode = MAX_EPISODE_FOR_ALPHA_MIN (for each state) so that
        # we can continue decreasing alpha further --from this self._alphas_at_max_episode value (for each state)--
        # with no lower bound --even if alpha_min has been specified-- past the MAX_EPISODE_FOR_ALPHA_MIN episode.
        self._alphas_at_max_episode = None
        # (Average) alpha used at each episode
        # Note: Using the average is only relevant when the adjustment is by state occupation count,
        # NOT when it is by episode; in fact, in the latter case, all the alphas are the same
        # for all states visited during the episode, so averaging over all states doesn't give any
        # different information than the alpha value that is common for all states.
        self.alpha_mean_by_episode = deque([])

        # Average reward observed at each episode
        self.average_reward_by_episode = deque([])

        # Time steps at which each episode ends
        self.times_at_episode_end = deque([])

        # State counts over ALL episodes run after reset, and state counts of just their first visits
        self._state_counts_over_all_episodes = np.zeros(self.env.getNumStates(), dtype=int)
        self._state_counts_first_visit_over_all_episodes = np.zeros(self.env.getNumStates(), dtype=int)

        # Instructions for resetting the value function
        self.reset_method = reset_method
        self.reset_params = reset_params
        self.reset_seed = reset_seed

        # Instructions about storing the trajectory
        self.store_history_over_all_episodes = store_history_over_all_episodes

    def reset(self, reset_episode=False, reset_value_functions=False, reset_average_reward=False):
        """
        Resets the variables that store information about the episode

        Parameters:
        reset_episode: (opt) bool
            Whether to reset the episode to the first one, which includes resetting
            all measures that are tracked by episode (e.g. alpha_mean_by_episode).
            default: False

        reset_value_functions: (opt) bool
            Whether to reset all the value functions to their initial estimates (e.g. all zeros or a random value).
            default: False

        reset_average_reward: (opt) bool
            Whether to reset the average reward its initial estimate (zero).
            default: False
        """
        if reset_episode:
            self._reset_at_start_of_first_episode()

        # Increase episode counter
        # (note that the very first episode run is #1 because reset() is called by __init__())
        self.episode += 1

        # Reset the attributes that keep track of states and rewards received during learning at the new episode that will start
        self._reset_at_start_of_episode()

        # Only reset the initial estimates of the value functions at the very first episode (the episode counter starts at 1)
        # (since each episode should leverage what the agent learned so far!)
        if self.episode == 1 or reset_value_functions or reset_average_reward:
            # Reset all the learning information by calling the super class reset, which is generic, i.e. it does NOT assume episodic tasks
            # Note that such super class calls the reset_value_function() method defined in the specific class that is inheriting from the super class!
            # So, in this case, it ends up calling the reset_value_functions() defined below!!
            super().reset(reset_learning_epoch=True, reset_alphas=True, reset_value_functions=reset_value_functions, reset_average_reward=reset_average_reward, reset_trajectory=True, reset_counts=True)

    def setParams(self, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min):
        self.alpha = alpha if alpha is not None else self.alpha
        self.adjust_alpha = adjust_alpha if adjust_alpha is not None else self.adjust_alpha
        self.alpha_update_type = alpha_update_type if alpha_update_type is not None else self.alpha_update_type
        self.adjust_alpha_by_episode = adjust_alpha_by_episode if adjust_alpha_by_episode is not None else self.adjust_alpha_by_episode
        self.alpha_min = alpha_min if alpha_min is not None else self.alpha_min

    def _reset_at_start_of_first_episode(self):
        "Resets the attributes that should be reset when the very first episode is run"
        # Episode number (this number will be incremented soon after the call to this method so that the first episode has number 1)
        self.episode = 0

        # Information about EACH episode run
        self.alpha_mean_by_episode = deque([])
        self.average_reward_by_episode = deque([])
        self.times_at_episode_end = deque([])
        del self._state_counts_over_all_episodes, self._state_counts_first_visit_over_all_episodes, self._alphas
        self._state_counts_over_all_episodes = np.zeros(self.env.getNumStates())
        self._state_counts_first_visit_over_all_episodes = np.zeros(self.env.getNumStates())

        # Learning rate
        self._alphas = self.alpha * np.ones(self.env.getNumStates())
        self._alphas_at_max_episode = None

    def _reset_at_start_of_episode(self):
        """
        Resets internal structures that store information about EACH episode
        (all attributes referring to the current episode should start with an underscore)
        """

        # States and actions visited in the current episode and the state count
        self._times = deque([])
        self._states = deque([])
        self._actions = deque([])
        self._state_counts = np.zeros(self.env.getNumStates())
        self._states_first_visit_time = np.nan * np.ones(self.env.getNumStates())

        # Store the _rewards obtained after each action
        # We initialize the _rewards with one element so that there is an INDEX match between the
        # list of visited states during a trajectory and the list of rewards observed during that trajectory,
        # i.e. if the states in an episode are [2, 3, 2, 6, 5], where `5` is the terminal state,
        # and rewards = [0, -1.0, 0.5, 2.0, 1.0], their values correspond to
        # states  S(t=0), S(t=1), ..., S(t=4) and
        # rewards R(t=0), R(t=1), ..., R(t=4), which also means that, given t,
        # S(t+1) is the state where the system goes after which the agent receives the reward R(t+1).
        # This also means that the sequence of state, action, reward as in
        # S(0), A(0), R(1), S(1), A(1), R(2), ...
        # can be retrieved by referring to the CORRESPONDING indices in the above lists, i.e. as
        # self._states[0], self._actions[0], self._rewards[1], self._states[1], self._actions[1], self._rewards[2], ...
        #
        # This also means that, when we want to look at the trajectory using the super class methods getTimes(), getStates(), getActions(), getRewards(),
        # we should retrieve the trajectory using:
        #   [(t, r, s, a) for t, r, s, a in zip(learner.getTimes(), learner.getRewards(), learner.getStates(), learner.getActions()]
        # because R(t) is the reward received when visiting state S(t) and A(t) is the action taken AFTER visiting S(t),
        # as explained in the DESCRIPTION of the class above.
        #
        # Note in addition that it is expected that the length of the list of states and the length of the list of rewards
        # be EQUAL *only* at the end of the episode (unless we are learning the value functions in a
        # CONTINUING learning task setting, even if the learning environment has terminal states --i.e. where naturally
        # an EPISODIC learning task is called for). In fact, while the episode is ongoing, the list of states
        # has one element less than the list of rewards, because the list of rewards is initialized as e.g. `[0]` here
        # while the list of states is initialized as `[]` (empty).
        # The particular learner being used is responsible of making sure that the list of states has the
        # same number of elements as the list of rewards ONCE THE EPISODE HAS ENDED, by calling e.g. the
        # store_trajectory_at_episode_end() method defined in this class.
        self._rewards = deque([self.env.getReward(self.env.getState())])

        # List of alphas used during the episode (which may vary from state to state and from the time visited)
        self._alphas_used_in_episode = deque([])

        # Average reward observed at each episode
        self._average_reward_in_episode = 0.0

        # Store the values of the next states at each time iteration
        # This is used for instance to check in mc.py the recursive calculation of G(t,lambda), i.e.:
        #     G(t,lambda) = R(t+1) + gamma * ( (1 - lambda) * V(S(t+1)) + lambda * G(t,lambda) )
        # stated in the paper "META-Learning state-based eligibility traces..." by Zhao et al. (2020)
        self._values_next_state = [np.nan]

    def reset_value_functions(self):
        """
        Resets the value functions stored in the object.

        Both the state- and the action-value function (V & Q) are tried to be reset,
        which are retrieved with the getV() and getQ() methods, respectively.
        These methods should be defined in the class inheriting from this class
        (see below the definition of getV() and getQ(), which raise a NonImplementedError,
        informing the reader that these methods are expected to be defined by inheriting classes).

        A warning is issued when one of the reset fails for a value function
        (either because the value function is not defined in the object, or because it is None, etc.).
        """
        try:
            self.getV().reset(method=self.reset_method, params_random=self.reset_params, seed=self.reset_seed)
        except Exception as e:
            warnings.warn(f"Resetting the value of the STATE value function failed. If this is needed, "
                          f"check whether the `getV()` is defined in the learner class '{self.__class__.__name__}', "
                          f"and if so, whether the reset() method is defined for the object containing the state value function.")
            print(e)
        try:
            self.getQ().reset(method=self.reset_method, params_random=self.reset_params, seed=self.reset_seed + 1 if self.reset_seed is not None else None)  # We sum +1 to the seed to avoid having the same initial values for V(s) and Q(s,a)
        except Exception as e:
            warnings.warn(f"Resetting the value of the ACTION value function failed. If this is needed, "
                          f"check whether the `getQ()` is defined in the learner class '{self.__class__.__name__}', "
                          f"and if so, whether the reset() method is defined for the object containing the action value function.")
            print(e)
        try:
            self.getA().reset(method=self.reset_method, params_random=self.reset_params, seed=self.reset_seed + 1 if self.reset_seed is not None else None)  # We sum +1 to the seed to avoid having the same initial values for V(s) and Q(s,a)
        except Exception as e:
            warnings.warn(f"Resetting the value of the ADVANTAGE function failed. If this is needed, "
                          f"check whether the `getA()` is defined in the learner class '{self.__class__.__name__}', "
                          f"and if so, whether the reset() method is defined for the object containing the action value function.")
            print(e)

    def _update_trajectory(self, t, state, action, reward):
        "Updates the trajectory of the CURRENT episode"
        self._times += [t]
        self._states += [state]
        self._actions += [action]
        self._rewards += [reward]
        # TODO: (2024/05/21) It would be good to, at some point, disentangle the update of the average reward from the trajectory update because they are two different concepts. However, this change will require important chnages in Learners code (e.g. learn() method) when calling update_trajectory().
        self._update_average_reward()

    def _update_state_counts(self, t, state):
        "Updates the count that keeps track of the state's first visit within the CURRENT episode"
        if self.env.isStateContinuous():
            # Discretize the state so that we can update the count of a visited state
            state = self.env.getIndexFromState(state)

        #print("t: {}, visit to state: {}".format(t, state))
        if np.isnan(self._states_first_visit_time[state]):
            self._state_counts_first_visit_over_all_episodes[state] += 1
            #print("\tFIRST VISIT!")
            #print("\tcounts first visit after: {}".format(self._state_counts_first_visit_over_all_episodes[state]))
            #print("\tall counts fv: {}".format(self._state_counts_first_visit_over_all_episodes))
        self._states_first_visit_time[state] = np.nanmin([t, self._states_first_visit_time[state]])

        # Keep track of state every-visit counts
        self._state_counts[state] += 1  # Counts per-episode
        self._state_counts_over_all_episodes[state] += 1  # Counts over all episodes

    def _update_average_reward(self):
        # TODO: (2023/08/31) According to the algorithm for learning the average reward presented in Sutton (2018), pag. 251, a better learner of the average reward uses a separate learning rate which is applied on the delta error... Implement this.
        # I think this should probably be implemented within each learner inheriting from this class...(?) because they store the delta error value to use.
        n_rewards_observed_so_far = len(self._rewards) - 1   # We subtract 1 to the length of self.rewards because the first element in self.rewards is a fictitious reward of 0 (see its initialization in the constructor)
        self._average_reward_in_episode += (self._rewards[-1] - self._average_reward_in_episode) / max(1, n_rewards_observed_so_far)    # `max(1, ...) to avoid division by 0 if we are storing the very first reward

    def _update_alphas(self, state):
        # with np.printoptions(precision=4):
        #    print("Before updating alpha: episode {}, state {}: state_count={:.0f}, alpha>={}: alpha={}\n{}" \
        #          .format(self.episode, state, self._state_counts_over_all_episodes[state], self.alpha_min, self.getAlphaForState(state), np.array(self._alphas)))
        if self.env.isStateContinuous():
            # Discretize the state so that we can update the count of a visited state
            state = self.env.getIndexFromState(state)

        # NOTE that we store the alpha value BEFORE its update, as this is the value that was used to learn prior to
        # updating alpha!
        self._alphas_used_in_episode += [self._alphas[state]]
        if self.adjust_alpha:
            if self.adjust_alpha_by_episode:
                # Update using the episode number (equal for all states)
                _time_divisor = self.func_adjust_alpha(max(1, self.episode - MIN_EPISODE + 2))
                self._alphas[state] = max(self.alpha_min, self.alpha / _time_divisor)
                    ## +2 => see the note below on the ELSE block for why we use +2 and not +1.
            else:
                if self.alpha_update_type == AlphaUpdateType.FIRST_STATE_VISIT:
                    state_count = self._state_counts_first_visit_over_all_episodes[state]
                else:
                    state_count = self._state_counts_over_all_episodes[state]
                _time_divisor = self.func_adjust_alpha(max(1, state_count - self.min_count_to_update_alpha + 2))
                    ## +2 => when state_count = min_count_to_update_alpha, the time divisor is > 1; if we used +1, the time divisor would be equal to 1
                    ## and this would imply that alpha would NOT be reduced, even if the state count had reached
                    ## the specified min count to adjust (reduce) alpha.
                # print("\t\tepisode: {}, state: {}, updating alpha... time divisor = {}".format(self.episode, state, time_divisor))
                # Update using the state occupation state_count over all past episodes
                if MAX_EPISODE_FOR_ALPHA_MIN is None or self.episode <= MAX_EPISODE_FOR_ALPHA_MIN:
                    # This means we should apply the alpha_min value as learning rate
                    # whenever its value is larger than the alpha value obtained by decreasing alpha
                    # by episode or by state count, at least until the episode number reaches the  MAX_EPISODE_FOR_ALPHA_MIN value,
                    # at which stage, alpha continues to decrease using the decreasing rule, starting off from the
                    # alpha value observed (for each state) at the MAX_EPISODE_FOR_ALPHA_MIN episode
                    # (which can be either alpha_min or larger than alpha_min).
                    self._alphas[state] = max(self.alpha_min, self.alpha / _time_divisor)
                    if MAX_EPISODE_FOR_ALPHA_MIN is not None and self.episode == MAX_EPISODE_FOR_ALPHA_MIN:
                        # Store the last alpha value observed for each state
                        # so that we can use it as starting point from now on when decreasing alpha further.
                        self._alphas_at_max_episode = self._alphas.copy()
                        # print("episode {}, state {}: alphas at max episode: {}".format(self.episode, state, self._alphas_at_max_episode))
                else:
                    # Start decreasing from the alpha value left at episode = MAX_EPISODE_FOR_ALPHA_MIN
                    # without any lower bound for alpha
                    self._alphas[state] = self._alphas_at_max_episode[state] / _time_divisor
                    # print("episode {}, state {}: alphas: {}".format(self.episode, state, self._alphas))

    def update_average_reward(self, T, state_end):
        """
        Updates the average reward over all episodes, when storing the history over all episodes, at the end of a new episode

        This is used when estimating the average reward of a continuing learning task.

        Arguments:
        T: int
            Length of the last episode, i.e. the number of steps that were taken until the episode ended.

        state_end: int
            Index of the state at which the last episode ended. Normally, this is only used for information purposes.
        """
        if self.store_history_over_all_episodes:
            # Update the average reward over all episodes because we are storing the history over all episodes,
            # therefore we might be interested in the average reward over all episodes.
            # Note that the update formula is the generalization of the usual update formula with just one new value
            # with the difference that here the update comes from T newly observed rewards (as opposed to 1),
            # i.e. what we need to sum to the current estimate of the average reward is \sum{t=1}{T} R(t)
            # (and we note that the average reward over E episodes is defined as: [ \sum{e=1}{E} \sum{t=1}{T} R_e(t) ] / \sum{e=1}{E} T_e
            # where T_e is the length of episode e).
            # Deducing the update formula is a little bit trickier than in the one-new-reward-observation case but it is perfectly doable --I just did it!
            # Therefore the update formula becomes:
            #   average <- average + (new_average_value - average) * (T + 1) / (sample_size_for(average) + T + 1)
            # where T + 1 is the length of the last observed episode + one step that corresponds to the step of going from the episode end state to the start state of the next
            # episode. This step should be taken into account when computing the average reward because the average reward is associated to a continuing learning task,
            # and when we are working on an environment that naturally calls for an EPISODIC learning task --e.g. a labyrinth where we look for the shortest path to the exit--
            # we define the CONTINUING learning task on which the average reward is computed by taking a final step of going from the episode end state to an episode start state.
            # And that is why we sum +1 to T as the sample size behind the `new_average_value`.
            # Note in addition, that the `new_average_value` to consider in the formula is the average reward observed in the episode but computed on (T+1) steps, meaning that
            # it is computed as:
            #   new_average_value = average_reward_observed_in_episode * T / (T+1)
            # which uses the formula that converts the episodic average reward to the continuing average reward, WHICH ASSUMES THAT THERE IS NO REWARD OBSERVED WHEN GOING FROM
            # THE EPISODE END STATE TO AN EPISODE START STATE (which is usually the case and reasonable, anyway).
            #
            # Note also that, in the tests run in test_estimators_discretetime.py we observe that the average reward computed like this
            # is very close to the average reward computed from cycles.
            #
            # Note finally that the attribute updated by this call to setAverageReward() is an attribute of the super class
            # (which is a generic learner, i.e. not only for learners on episodic tasks)

            # (2023/12/18) Update the OVERALL average reward (i.e. over ALL episodes) ITERATIVELY, i.e. using the latest observed average reward (in the latest episode)
            # Note that this happens ONLY when the length of the episode T is > 0 (see multiplier `int(T>0)` below), o.w. there is no episode contributing to the average reward!
            # IMPORTANT 1: This computation adjusts an episodic average to a continuing average, which has ONE MORE STEP in the trajectory, namely the start state,
            # which is NOT counted in the episodic average (because the rewards happen when the environment CHANGES from one state to the next,
            # meaning that the first average seen by the episodic learning agent is the reward observed when going from the start state to the next state
            # (and this in turn means that no reward is counted when going to the start state, because this "going to the start state" NEVER happens in an episodic learning task,
            # it only happens in a continuing learning task (whose average reward we are interested here in computing)).
            # IMPORTANT 2: This adjustment of the episodic average to the continuing average assumes that going to the start state does NOT give any reward!!
            # (this is usually the case and reasonable, but it is not really generic)
            # TODO: (2023/12/18) Fix the adjustment performed of the episodic average reward to a continuing average reward to the cases where a non-zero reward is perceived when transitioning to *a* start state (see "IMPORTANT 2" note written above)
            # NOTE that this may NOT be needed if we generalize the learning of value functions to the MC and TD(lambda) learners using the proposed method described in today's entry at Tasks-Projects.xlsx, where we would have only ONE episode on which the episodic average reward is computed.
            sample_size_for_current_estimated_average = self.sample_size_initial_reward_stored_in_learner + np.sum(self.times_at_episode_end) + len(self.times_at_episode_end)
                ## NOTE: (2024/04/17) Explanation of the use of `len(self.times_at_episode_end)` which I haven't explained so far, but needs an explanation:
                ## it is due to what is explained above about the sample size behind the EPISODIC average reward (which is T)
                ## and the sample size behind the CONTINUING average reward (T+1), i.e. each episode length stored in self.times_at_episode_end contains the EPISODIC sample size
                ## (as opposed to the CONTINUING sample size).
                ## Therefore we need to sum +1 to EACH episode, thus we obtain `len(self.times_at_episode_end)` resulting from summing +1 `len(self.times_at_episode_end)` times.
            updated_average = self.getAverageReward() + int(T>0) * (self._average_reward_in_episode * T / (T+1) - self.getAverageReward()) \
                                                                 * (T + 1) / (sample_size_for_current_estimated_average + (T + 1))
            # (2023/12/18) The following is a check that the updated average is equal to the regular average computed on all the rewards seen so far over ALL episodes
            # (Note: at some point, the print() below gives an error that something is an int and not a list, but I haven't figured out what the problem is, so I commented out, because this step is not crucial for the functioning of the process)
            #all_rewards_so_far = self.rewards + self._rewards
            #print(f"\nT={T}, state={state_end}\nEpisode average = {self._average_reward_in_episode}\nPrevious average = {self.getAverageReward()}\nEpisode length + 1 = {T+1}\nUpdated average = {updated_average} vs.\nComputed average = {np.mean(all_rewards_so_far)}")
            self.setAverageReward(updated_average)

            # Check that the updated average reward is correctly calculated (by comparing to the regular average over all the rewards observed over ALL episodes)
            # NOTE: This assumes that the store_trajectory_at_episode_end() method has not been called yet
            # (because we assume that the self.rewards attribute has not been updated with the newly observed self._rewards in the latest episode
            # TODO: (2024/01/03) Re-establish the assertion done here once I make it work again for the FV learning in the DISCOUNTED setting (using _run_simulation_fv_iterate())
            #all_rewards_so_far = self.rewards + deque([self._rewards])     # IMPORTANT: The concatenation of the rewards should be done in the same way it is done in store_trajectory_at_episode_end() when updating self.rewards
            # The problem comes up because of the parallel simulations that are run together with the FV particles in order to learn the value of the states inside the absorption set A (see the envs_normal environments in discrete.Simulator._run_simulation_fv_iterate()
            #assert np.isclose(updated_average, np.mean(np.concatenate(all_rewards_so_far)))
            # Regular average (over the whole history of stored rewards, as opposed to doing an iterated update)
            # (2023/12/18) This assumes that the self.rewards list has been updated BEFORE calling this method, which currently is NOT the case,
            # because e.g. the MC learner learn_mc() learns (i.e. calls this method) BEFORE storing the trajectory (by calling store_trajectory_at_episode_end(),
            # which is the method responsible for storing the newly observed rewards at the latest episode in GenericLearner's attribute `self.rewards`.
            #self.setAverageReward(np.mean(np.concatenate(self.rewards)))

    def store_trajectory_at_episode_end(self, T, state_end, action=np.nan, reward=0.0, debug=False):
        """
        Stores the trajectory (in the super class attributes) observed during the current episode assuming the episode has finished.
        The trajectory is either REPLACED or ADDED to any previously existing trajectory already stored in the attributes of the super class
        (typically super.states, super.actions, super.rewards), depending on attribute store_history_over_all_episodes.

        The method also computes and stores the following averages:
        - observed average reward by episode
        - average alpha (over all states) used by episode
        Note that the average alpha by episode requires that the alphas (over all states) used in the episode
        should have already been stored in the object's attribute (typically self._alphas_used_by_episode)
        prior to calling this method. Otherwise, the average alpha stored in the object's attribute
        (typically self.alpha_mean_by_episode) will have all NaN values.

        Notes:
        1) This method should be called by the learner inheriting from this class at the end of the episode
        so that the length of the lists storing the states, actions and rewards observed during the episode
        all have the same length (at the end of the episode).

        2) Once the above is the case, each index in the states list and in the rewards list correspond to the SAME
        time index (i.e. self.states[t], self.rewards[t] correspond respectively to S(t) and R(t), meaning
        that R(t) is the reward received AFTER the system visits state S(t) --recall that R(0) has been
        set to 0 at the beginning of the episode (see definition of self._rewards() in the
        _reset_at_start_of_episode() method of this class).

        3) In addition, action self.actions[t] contains action A(t), i.e. the action taken AFTER visiting state S(t).
        The last action of the episode, i.e. A(T), is set to np.nan when this method is called,
        because when the episode ends, no action is taken at the state where the system ends up.

        Arguments:
        T: int
            Time at which the episode ends.

        state_end: int
            Trajectory's end state that is added to the trajectory information.

        debug: (opt) bool
            Whether to show the end state and the time at which the episode ends, in addition to
            the distribution of the latest alpha value by state.
            default: False
        """
        if debug:
            print("Episode {}: End state = {} at time step {}".format(self.episode, state_end, T))
            self.plot_alphas(T)

        # Store information that we store for each episode
        self.alpha_mean_by_episode += [np.mean(self._alphas_used_in_episode)]   # This is an average over all environment states
        self.average_reward_by_episode += [self._average_reward_in_episode]
        self.times_at_episode_end += [T]

        # Add the end state to the trajectory
        self._times += [T]
        self._states += [state_end]
        self._actions += [action]
        # TODO: (2024/02/08) Uncomment this when we are ready to compute the average reward associated to the learning task, either EPISODIC or CONTINUING: currently the computed average reward by self.update_average_reward() is ALWAYS the CONTINUING average reward, even if the learning task is EPISODIC!
        # When we do this change, the method self.update_average_reward() will have to be changed by removing the ratio `T / (T+1)` which is now used to adjust the would be episodic average reward to the continuing average reward.
        # Once we implement the episodic average reward calculation (when the learning task is EPISODIC) we would not need to do any adjustment, because the original formula of the iterative update of the average reward
        # (described in the comment above the update formula in self.update_average_reward()) will already be the correct formula. And note that this will ALSO solve the problem mentioned in
        # self.update_average_reward that using the ratio T/(T+1) assumes that the last reward observed in the continuing context (i.e. the reward of going from a terminal state to a start state) is 0
        # (as that reward is going to be stored in self._rewards precisely by the line that will be uncommented right here!)
        #if self.task == LearningTask.CONTINUING:
        #    self._rewards += [reward]

        # Assign the new trajectory observed in the current episode
        # NOTE: All the following attributes on the LHS are attributes of the superclass!
        if self.store_history_over_all_episodes:
            self.times += [self._times.copy()]
            self.states += [self._states.copy()]
            self.actions += [self._actions.copy()]
            self.rewards += [self._rewards.copy()]
        else:
            self.times = self._times.copy()
            self.states = self._states.copy()
            self.actions = self._actions.copy()
            self.rewards = self._rewards.copy()

    def plot_alphas(self, t):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.env.all_states, self._alphas, 'g.-')
        ax = plt.gca()
        ax.set_xlabel("state")
        ax.set_ylabel("alpha")
        ax.set_title("Learning rate (alpha) for each state (episode {}, t={})".format(self.episode, t))
        ax2 = ax.twinx()  # Create a secondary axis sharing the same x axis
        ax2.bar(self.env.all_states, self._state_counts_over_all_episodes, color="blue", alpha=0.3)
        plt.sca(ax)  # Go back to the primary axis

    def getStateCounts(self, first_visit=False):
        "Returns the array of state counts, either the first-visit state counts or the every-visit state counts"
        if first_visit:
            return self._state_counts_first_visit_over_all_episodes
        else:
            return self._state_counts_over_all_episodes

    def getAlphasByState(self):
        return self._alphas

    def getAlphaForState(self, state):
        return self._alphas[state]

    def getAverageAlphaByEpisode(self):
        return self.alpha_mean_by_episode

    def getAverageRewardByEpisode(self):
        return self.average_reward_by_episode

    def getTimesAtWhichEachEpisodeEnded(self):
        return self.times_at_episode_end

    def getV(self):
        "Returns the object containing the state value function estimation"
        # This method is not implemented because the subclass implementing the actual learner may define the state value function differently (e.g. using different attribute names)
        # Also, this method is required because it is called by the reset_value_functions() method defined in this class.
        raise NotImplementedError

    def getQ(self):
        "Returns the object containing action value function estimation"
        # This method is not implemented because the subclass implementing the actual learner may define the action value function differently (e.g. using different attribute names)
        # Also, this method is required because it is called by the reset_value_functions() method defined in this class.
        raise NotImplementedError

    def getA(self):
        "Returns the object containing advantage function estimation"
        # This method is not implemented because the subclass implementing the actual learner may define the action value function differently (e.g. using different attribute names)
        # Also, this method is required because it is called by the reset_value_functions() method defined in this class.
        raise NotImplementedError

    def setSampleSizeForAverageReward(self):
        """
        Sets the sample size behind the calculation of the average reward based on the information stored in the self.times_at_episode_end
        which stores the length of each episode (at the end of which the average reward is updated).
        This information can be used to update the average reward from a previous estimate in subsequent learning moments carried out with the same learner.
        """
        sample_size = np.sum(self.times_at_episode_end)
        super().setSampleSizeForAverageReward(sample_size)
