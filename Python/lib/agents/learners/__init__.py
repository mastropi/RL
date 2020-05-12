# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:15:04 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all learners.

All learner classes defined in the current package should:
a) Implement the following attributes:
    - env --> the environment on which the learning takes place
    - V --> an object containing the information on how the state value function is ESTIMATED.
    - Q --> an object containing the information on how the state-action value function is ESTIMATED.
    - alpha --> the learning rate
    - gamma --> the reward discount parameter
b) Implement the following methods:
All learners are assumed to have the following methods defined:
    - reset() --> resets the state of the learner to start learning anew
        (e.g. all the estimates of the value functions are reset to 0)
    - setParams() --> set the parameters of the learner (if any)
    - learn_pred_V() --> prediction problem: learns the state value function under the current policy (V(s))
    - learn_pred_Q() --> prediction problem: learns the action-value function under the currenty policy (Q(s,a))
    - learn_ctrl_policy() --> control problem: learns the optimal policy

VALUE FUNCTIONS:
The value functions used by the learner classes are assumed to:
a) Be defined in terms of weights, e.g. the state value function for state s is V(s,w),
where w is the vector of weights applied to a set of features X.
Note that this assumption does NOT offer any limitation, since a tabular value function
can be defined using binary/dummy features. 

b) Have the following methods defined:
- reset(): resets the vector w of weigths to their initial estimates 
- getWeights(): reads the vector w of weights
- setWeights(): updates the vector w of weights
- getValue(): reads the value function for a particular state or state-action
- getValues(): reads the value function for ALL states or state-actions
"""

from enum import Enum, unique

import numpy as np

from Python.lib.environments import EnvironmentDiscrete


MIN_COUNT = 1      # Minimum state count to start shrinking alpha
MIN_EPISODE = 1    # Minimum episode count to start shrinking alpha
MAX_EPISODE_FOR_ALPHA_MIN = None  # Maximum episode on which the minimum alpha value above is applied

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2


class Learner:
    """
    Class defining methods that are generic to ALL learners.
    
    NOTE: Before using any learner the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: one when the learner is constructed
    and one prior to the first simulation, making e.g. the episode method be equal to 2 at
    the first simulation as opposed to the correct value 1.
    """

    def __init__(self, env, alpha,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT, adjust_alpha_by_episode=True,
                 alpha_min=0.):
        """
        Parameters:
        env: EnvironmentDiscrete
            Environment where the agent learns.
        
        alpha: float
            Learning rate.
        
        alpha_update_type: LearnerType
            Type of learner. E.g. LearnerType.TD, LearnerType.MC.
        """
        if not isinstance(env, EnvironmentDiscrete):
            raise TypeError("The environment must be of type {} from the {} module ({})" \
                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        if not isinstance(alpha_update_type, AlphaUpdateType):
            raise TypeError("The alpha_update_type of learner must be of type {} from the {} module ({})" \
                            .format(AlphaUpdateType.__name__, AlphaUpdateType.__module__, alpha_update_type.__class__))

        self.env = env
        self.alpha = alpha
        self.adjust_alpha = adjust_alpha
        self.alpha_update_type = alpha_update_type
        self.adjust_alpha_by_episode = adjust_alpha_by_episode
        self.alpha_min = alpha_min          # Used when adjust_alpha=True

        # Information of the observed trajectory at the END of the episode
        # (so that it can be retrieved by the user if needed as a piece of information)
        self.states = []
        self.rewards = []

        # Episode counter
        self.episode = 0
        # (Average) alpha used at each episode
        self.alpha_mean_by_episode = []

        # State counts over ALL episodes run after reset
        self._state_counts_overall = np.zeros(self.env.getNumStates())

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
            self.episode = 0
            self.alpha_mean_by_episode = []
            # State counts over ALL episodes run after reset
            self._state_counts_overall = np.zeros(self.env.getNumStates())
            # State counts of first visits over ALL episodes run after reset 
            self._state_counts_first_visit_overall = np.zeros(self.env.getNumStates())
            # Learning rate for each state (so that we can reduce alpha with time)
            self._alphas = self.alpha * np.ones(self.env.getNumStates())
            # Learning rate at episode = MAX_EPISODE_FOR_ALPHA_MIN so that we can continuing applying a non-bounded alpha
            self._alphas_at_max_episode = None

        # Increase episode counter
        # (note that the very first episode run is #1 because reset() is called by __init__()) 
        self.episode += 1
        # List of alphas used during the episode (which may vary from state to state and from the time visited)
        self._alphas_used_in_episode = []

        # Reset the attributes that keep track of states and rewards received during learning
        self._reset_at_start_of_episode()

        # Only reset the initial estimates of the value functions at the very first episode
        # (since each episode should leverage what the agent learned so far!)
        if self.episode == 1 or reset_value_functions:
            self.V.reset()

    def setParams(self, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min):
        self.alpha = alpha if alpha is not None else self.alpha
        self.adjust_alpha = adjust_alpha if adjust_alpha is not None else self.adjust_alpha
        self.alpha_update_type = alpha_update_type if alpha_update_type is not None else self.alpha_update_type
        self.adjust_alpha_by_episode = adjust_alpha_by_episode if adjust_alpha_by_episode is not None else self.adjust_alpha_by_episode
        self.alpha_min = alpha_min if alpha_min is not None else self.alpha_min

    def _reset_at_start_of_episode(self):
        """
        Resets internal structures used during learning in each episode
        (all attributes reset here should start with an underscore, i.e. they should be private)
        """

        # Store the _states visited in the episode and their count
        self._states = []
        self._state_counts = np.zeros(self.env.getNumStates())
        self._states_first_visit_time = np.nan*np.ones(self.env.getNumStates())

        # Store the _rewards obtained after each action
        # We initialize the _rewards with one element equal to 0 for ease of notation
        # when retrieving the state and the reward received at each time t
        # because the state is defined for t = 0, ..., T-1 
        # while the reward is defined for t = 1, ..., T
        self._rewards = [0]

    def _update_trajectory(self, t, state, reward):
        "Updates the trajectory and related information based on the current time, current state and the observed reward"
        self._states += [state]
        self._rewards += [reward]

        # Keep track of state first visits
        #print("t: {}, visit to state: {}".format(t, state))
        if np.isnan(self._states_first_visit_time[state]):
            self._state_counts_first_visit_overall[state] += 1
            #print("\tFIRST VISIT!")
            #print("\tcounts first visit after: {}".format(self._state_counts_first_visit_overall[state]))
            #print("\tall counts fv: {}".format(self._state_counts_first_visit_overall))
        self._states_first_visit_time[state] = np.nanmin([t, self._states_first_visit_time[state]])

        # Keep track of state every-visit counts
        self._state_counts[state] += 1              # Counts per-episode
        self._state_counts_overall[state] += 1      # Counts over all episodes

    def _update_alphas(self, state):
        self._alphas_used_in_episode += [self._alphas[state]]
        if self.adjust_alpha:
            if self.adjust_alpha_by_episode:
                # Update using the episode number (equal for all states)
                self._alphas[state] =  np.max([ self.alpha_min, self.alpha / np.max([1, self.episode - MIN_EPISODE + 1]) ])
            else:
                if self.alpha_update_type == AlphaUpdateType.FIRST_STATE_VISIT:
                    state_count = self._state_counts_first_visit_overall[state]
                else:
                    state_count = self._state_counts_overall[state]
                time_divisor = np.max([1, state_count - MIN_COUNT + 2]) # +2 => when state_count = MIN_COUNT, time_divisor is > 1, o.w. alpha would not be changed
                #print("\t\tepisode: {}, state: {}, updating alpha... time divisor = {}".format(self.episode, state, time_divisor))
                # Update using the state occupation state_count over all past episodes 
                if MAX_EPISODE_FOR_ALPHA_MIN is None or self.episode <= MAX_EPISODE_FOR_ALPHA_MIN:
                    # This means we should apply the alpha_min value indefinitely or
                    # up to the max episode specified by MAX_EPISODE_FOR_ALPHA_MIN
                    self._alphas[state] =  np.max([ self.alpha_min, self.alpha / time_divisor ])
                    if MAX_EPISODE_FOR_ALPHA_MIN is not None and self.episode == MAX_EPISODE_FOR_ALPHA_MIN:
                        # Store the last alpha value observed for each state
                        # so that we can use it as starting point from now on.
                        self._alphas_at_max_episode = self._alphas.copy()
                        #print("episode {}, state {}: alphas at max episode: {}".format(self.episode, state, self._alphas_at_max_episode))
                else:
                    # Start decreasing from the alpha value left at episode = MAX_EPISODE_FOR_ALPHA_MIN
                    # without any lower bound for alpha
                    self._alphas[state] = self._alphas_at_max_episode[state] / time_divisor
                    #print("episode {}, state {}: alphas: {}".format(self.episode, state, self._alphas))

        assert set([state]).issubset( set(self.env.terminal_states) ) == False, \
                "The state on which the alpha is computed is NOT a terminal state"

        #print("episode {}, state {}: state_count={:.0f}, alphas >= {}: {}     {}" \
        #      .format(self.episode, state, self._state_counts_overall[state], self.alpha_min, self._alphas[state], self._alphas))

    def store_trajectory(self):
        "Stores the trajectory observed during the episode"
        # Remove any trajectory stored from a previous run from memory
        del self.states
        del self.rewards
        # Assign the new trajectory observed in the current episode 
        self.states = self._states.copy()
        self.rewards = self._rewards.copy()

    def final_report(self, T):
        "Processes to be run at the end of the learning process (e.g. at the end of the episode)"
        # Store the (average) learning rate used during the episode for the new episode just ended
        # Note: Using the average is only relevant when the adjustment is by state occupation count,
        # NOT when it is by episode since in the latter case all the alphas are the same
        # for all states visited during the episode.
        self.alpha_mean_by_episode += [np.mean(self._alphas_used_in_episode)]

        if self.debug:
            self.plot_alphas(T)

    def plot_alphas(self, t):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.env.all_states, self._alphas, 'g.-')
        ax = plt.gca()
        ax.set_xlabel("state")
        ax.set_ylabel("alpha")
        ax.set_title("Learning rate (alpha) for each state (episode {}, t={})".format(self.episode, t))
        ax2 = ax.twinx()    # Create a secondary axis sharing the same x axis
        ax2.bar(self.env.all_states, self._state_counts_overall, color="blue", alpha=0.3)
        plt.sca(ax) # Go back to the primary axis

    def getV(self):
        "Returns the object containing information about the state value function estimation"
        return self.V

    def getQ(self):
        "Returns the object containing information about the state-action value function estimation"
        return self.Q
