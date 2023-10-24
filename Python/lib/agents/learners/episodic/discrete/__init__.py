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
    - gamma --> the reward discount parameter
b) COULD implement the following methods:
    - reset() --> resets the state of the learner to start learning anew
        (e.g. all the estimates of the value functions are reset to 0)
    - setParams() --> set the parameters of the learner (if any)
    - learn_pred_V() --> prediction problem: learns the state value function under the current policy (V(s))
    - learn_pred_Q() --> prediction problem: learns the action-value function under the current policy (Q(s,a))
    - learn_ctrl_policy() --> control problem: learns the optimal policy
    - getStateCounts(first_visit) --> returns the state counts over all run episodes, optionally the first visit counts.
    - getV() --> returns the state value function
    - getQ() --> returns the state-action value function
"""

import warnings
from enum import Enum, unique

import numpy as np

from Python.lib.environments import EnvironmentDiscrete
from Python.lib.agents.learners import GenericLearner, LearningCriterion, ResetMethod

MIN_COUNT = 1  # Minimum state count to start shrinking alpha
MIN_EPISODE = 1  # Minimum episode count to start shrinking alpha
MAX_EPISODE_FOR_ALPHA_MIN = None  # Maximum episode on which the minimum alpha value above is applied


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
    """
    def __init__(self, env, criterion: LearningCriterion=LearningCriterion.DISCOUNTED, alpha: float=1.0,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT, adjust_alpha_by_episode=False,
                 func_adjust_alpha=None,
                 alpha_min=0.,
                 min_count_to_update_alpha=0, min_time_to_update_alpha=0,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False):
        """
        Parameters:
        env: EnvironmentDiscrete
            Environment where learning takes place.
            It should have method getNumStates() defined, retrieving the number of fixed states in the environment.

        criterion: (opt) LearningCriterion
            See documentation of the super class.
            default: LearningCriterion.DISCOUNTED

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
            Minimum count of a state-action pair at which alpha starts to be updated by the update_learning_rate_by_state_action_count(s,a) method.

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
        super().__init__(env, criterion=criterion, alpha=alpha, adjust_alpha=adjust_alpha, func_adjust_alpha=func_adjust_alpha, alpha_min=alpha_min,
                         min_count_to_update_alpha=min_count_to_update_alpha, min_time_to_update_alpha=min_time_to_update_alpha)
        self.alpha_update_type = alpha_update_type
        self.adjust_alpha_by_episode = adjust_alpha_by_episode

        # Episode counter
        self.episode = 0

        # Current learning rate across states
        self._alphas = self.alpha * np.ones(self.env.getNumStates())
        # Learning rate at episode = MAX_EPISODE_FOR_ALPHA_MIN so that we can continue applying a non-bounded alpha
        self._alphas_at_max_episode = None
        # (Average) alpha used at each episode
        # Note: Using the average is only relevant when the adjustment is by state occupation count,
        # NOT when it is by episode; in fact, in the latter case, all the alphas are the same
        # for all states visited during the episode, so averaging over all states doesn't give any
        # different information than the alpha value that is common for all states.
        self.alpha_mean_by_episode = []

        # Average reward observed at each episode
        self.average_reward_by_episode = []

        # Time steps at which each episode ends
        self.times_at_episode_end = []

        # State counts over ALL episodes run after reset, and state counts of just their first visits
        self._state_counts_over_all_episodes = np.zeros(self.env.getNumStates())
        self._state_counts_first_visit_over_all_episodes = np.zeros(self.env.getNumStates())

        # Instructions for resetting the value function
        self.reset_method = reset_method
        self.reset_params = reset_params
        self.reset_seed = reset_seed

        # Instructions about storing the trajectory
        self.store_history_over_all_episodes = store_history_over_all_episodes

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
            self._reset_at_start_of_first_episode()

        # Increase episode counter
        # (note that the very first episode run is #1 because reset() is called by __init__())
        self.episode += 1

        # Reset the attributes that keep track of states and rewards received during learning at the new episode that will start
        self._reset_at_start_of_episode()

        # Only reset the initial estimates of the value functions at the very first episode
        # (since each episode should leverage what the agent learned so far!)
        if self.episode == 1 or reset_value_functions:
            # Reset all the learning information by calling the super class reset, which is generic, i.e. it does NOT assume episodic tasks
            # Note that such super class calls the reset_value_function() method defined in the specific class that is inheriting from the super class!
            # So, in this case, it ends up calling the reset_value_functions() defined below!!
            super().reset(reset_learning_epoch=True, reset_alphas=True, reset_value_functions=True, reset_trajectory=True, reset_counts=True)

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
        self.alpha_mean_by_episode = []
        self.average_reward_by_episode = []
        self.times_at_episode_end = []
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
        self._states = []
        self._actions = []
        self._state_counts = np.zeros(self.env.getNumStates())
        self._states_first_visit_time = np.nan * np.ones(self.env.getNumStates())

        # Store the _rewards obtained after each action
        # We initialize the _rewards with one element equal to 0 so that there is an INDEX match between the
        # list of visited states during a trajectory and the list of rewards observed during that trajectory,
        # i.e. if the states in an episode are [2, 3, 2, 6, 5], where `5` is the terminal state,
        # and rewards = [0, -1.0, 0.5, 2.0, 1.0], their values correspond to states S(t=0), S(t=1), ..., S(t=4)
        # and R(t=0) (= 0 ALWAYS because it is defined here as such), R(t=1), ..., R(t=4),
        # which also means that, given t, S(t+1) is the state where the system goes after which the agent
        # receives the reward R(t+1).
        # Note that it is expected that the length of the list of states and the length of the list of rewards
        # be EQUAL *only* at the end of the episode as, while the episode is ongoing, the list of states
        # would have one element less than the list of rewards (because the list of rewards is initialized
        # as `[0]` here, while the list of states is initialized as `[]` (empty).
        # The particular learner being used is responsible of making sure that the list of states has the
        # same number of elements as the list of rewards ONCE THE EPISODE HAS ENDED.
        self._rewards = [0]

        # List of alphas used during the episode (which may vary from state to state and from the time visited)
        self._alphas_used_in_episode = []

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
        except:
            warnings.warn(f"Resetting the value of the STATE value function failed. If this is needed, "
                          f"check whether the `getV()` is defined in the learner class '{self.__class__.__name__}', "
                          f"and if so, whether the reset() method is defined for the object containing the state value function.")
        try:
            self.getQ().reset(method=self.reset_method, params_random=self.reset_params, seed=self.reset_seed)
        except:
            warnings.warn(f"Resetting the value of the ACTION value function failed. If this is needed, "
                          f"check whether the `getQ()` is defined in the learner class '{self.__class__.__name__}', "
                          f"and if so, whether the reset() method is defined for the object containing the action value function.")

    def _update_trajectory(self, t, state, action, reward):
        "Updates the trajectory of the CURRENT episode"
        self._states += [state]
        self._actions += [action]
        self._rewards += [reward]
        self._update_state_counts(t, state)
        self._update_average_reward()

    def _update_state_counts(self, t, state):
        "Updates the count that keeps track of the state's first visit within the CURRENT episode"
        # print("t: {}, visit to state: {}".format(t, state))
        if np.isnan(self._states_first_visit_time[state]):
            self._state_counts_first_visit_over_all_episodes[state] += 1
            # print("\tFIRST VISIT!")
            # print("\tcounts first visit after: {}".format(self._state_counts_first_visit_over_all_episodes[state]))
            # print("\tall counts fv: {}".format(self._state_counts_first_visit_over_all_episodes))
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
        # TODO: (2023/09/06) The following assertion was commented out when I was implementing the FV estimator on discrete environments (e.g. labyrinth) because the system can be in a terminal state when updating alpha (e.g. the labyrinth's exit)... I don't think updating alpha in a terminal state is wrong... is it?
        #assert not self.env.isTerminalState(state), \
        #    "The state on which alpha is computed must NOT be a terminal state ({})".format(state)
        # with np.printoptions(precision=4):
        #    print("Before updating alpha: episode {}, state {}: state_count={:.0f}, alpha>={}: alpha={}\n{}" \
        #          .format(self.episode, state, self._state_counts_over_all_episodes[state], self.alpha_min, self._alphas[state], np.array(self._alphas)))

        # NOTE that we store the alpha value BEFORE its update, as this is the value that was used to learn prior to
        # updating alpha!
        self._alphas_used_in_episode += [self._alphas[state]]
        if self.adjust_alpha:
            if self.adjust_alpha_by_episode:
                # Update using the episode number (equal for all states)
                self._alphas[state] = np.max([self.alpha_min, self.alpha / np.max([1, self.episode - MIN_EPISODE + 2])])
                ## +2 => when episode = MIN_EPISODE, time_divisor is > 1,
                ## o.w. alpha would not be changed for the next update iteration
            else:
                if self.alpha_update_type == AlphaUpdateType.FIRST_STATE_VISIT:
                    state_count = self._state_counts_first_visit_over_all_episodes[state]
                else:
                    state_count = self._state_counts_over_all_episodes[state]
                time_divisor = np.max([1, state_count - MIN_COUNT + 2])
                ## +2 => when state_count = MIN_COUNT, time_divisor is > 1,
                ## o.w. alpha would not be changed for the next update iteration
                # print("\t\tepisode: {}, state: {}, updating alpha... time divisor = {}".format(self.episode, state, time_divisor))
                # Update using the state occupation state_count over all past episodes
                if MAX_EPISODE_FOR_ALPHA_MIN is None or self.episode <= MAX_EPISODE_FOR_ALPHA_MIN:
                    # This means we should apply the alpha_min value indefinitely or
                    # up to the max episode specified by MAX_EPISODE_FOR_ALPHA_MIN
                    self._alphas[state] = np.max([self.alpha_min, self.alpha / time_divisor])
                    if MAX_EPISODE_FOR_ALPHA_MIN is not None and self.episode == MAX_EPISODE_FOR_ALPHA_MIN:
                        # Store the last alpha value observed for each state
                        # so that we can use it as starting point from now on.
                        self._alphas_at_max_episode = self._alphas.copy()
                        # print("episode {}, state {}: alphas at max episode: {}".format(self.episode, state, self._alphas_at_max_episode))
                else:
                    # Start decreasing from the alpha value left at episode = MAX_EPISODE_FOR_ALPHA_MIN
                    # without any lower bound for alpha
                    self._alphas[state] = self._alphas_at_max_episode[state] / time_divisor
                    # print("episode {}, state {}: alphas: {}".format(self.episode, state, self._alphas))

    def store_trajectory_at_end_of_episode(self, T, state, debug=False):
        """
        Stores the trajectory (in the super class attributes) observed during the current episode assuming the episode has finished.
        The trajectory is either REPLACED or ADDED to any previously existing trajectory already stored in the attributes of the super class
        (typically super.states, super.actions, super.rewards), depending on attribute store_history_over_all_episodes.

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
        because when the episode ends, no action is taken at the final state where the system ends up.

        Arguments:
        T: int
            Time at which the episode ends.

        state: int
            Final state of the trajectory that is added to the trajectory information.

        debug: (opt) bool
            Whether to show the terminal state and the time at which the episode ends, in addition to
            the distribution of the latest alpha value by environment state.
            default: False
        """
        if debug:
            print("Episode {}: Terminal state = {} at time step {}".format(self.episode, state, T))
            self.plot_alphas(T)

        # Store information that we store for each episode
        self.alpha_mean_by_episode += [np.mean(self._alphas_used_in_episode)]   # This is an average over all environment states
        self.average_reward_by_episode += [self._average_reward_in_episode]
        self.times_at_episode_end += [T]

        # Add the terminal state to the trajectory
        self._states += [state]
        self._actions += [np.nan]

        # Assign the new trajectory observed in the current episode
        if self.store_history_over_all_episodes:
            self.states += [self._states.copy()]
            self.actions += [self._actions.copy()]
            self.rewards += [self._rewards.copy()]
            # Update the average reward over all episodes because we are storing the history over all episodes,
            # therefore we might be interested in the average reward over all episodes.
            # Note that the update formula is the generalization of the usual update formula with just one new value
            # with the difference that here the update comes from T newly observed rewards (as opposed to 1),
            # therefore the update formula becomes:
            #   average <- average + (new_average_value - average) * T / (sample_size_for(average) + T)
            # Note that this information is an attribute of the super class (which is a generic learner, i.e. not only for learners on episodic tasks)
            self.setAverageReward(self.getAverageReward() + (self._average_reward_in_episode - self.getAverageReward()) * self.times_at_episode_end[-1] / np.sum(self.times_at_episode_end))
        else:
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

    def getAverageAlphaByEpisode(self):
        return self.alpha_mean_by_episode

    def getAverageRewardByEpisode(self):
        return self.average_reward_by_episode

    def getTimesAtWhichEachEpisodeEnded(self):
        return self.times_at_episode_end

    def getV(self):
        "Returns the object containing the state value function estimation"
        raise NotImplementedError

    def getQ(self):
        "Returns the object containing action value function estimation"
        raise NotImplementedError
