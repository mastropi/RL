# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropietro
@description: Definition of Temporal Difference algorithms.
"""


from enum import Enum, unique

import numpy as np
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners import LearningCriterion, LearningTask, LearningMode, ResetMethod
from Python.lib.agents.learners.episodic.discrete import Learner, AlphaUpdateType

from Python.lib.utils.basic import set_numpy_options, reset_numpy_options
import Python.lib.utils.plotting as plotting


@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AdaptiveLambdaType(Enum):
    ATD = 1     # (full) Adaptive TD(lambda)
    HATD = 2    # Homogeneously Adaptive TD(lambda)


class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Arguments:
    env: Environment (e.g. gym.envs.toy_text.discrete.DiscreteEnv)
        The environment where the learning takes place.
        It must have the following methods defined:
        - getNumActions()
        - getNumStates()
        - getAllStates()

    store_history_over_all_episodes: (opt) bool
        Whether to store in the attributes of the generic super class storing the trajectory
        (e.g. states, actions, rewards) the whole trajectory history, over all episodes.
        If this is requested, the trajectory is stored in lists of lists, where each sublist is
        the trajectory observed in an episode.
        This is useful if we need to compute something using the whole trajectory history or whether
        we want to know what the trajectories were on all episodes.
        This value is set to True if the learning criterion is the average reward because we need to
        access the average reward value estimated over ALL the episodes run as the average reward
        criterion is used on continuing learning tasks.
        default: False
    """

    def __init__(self, env,
                 dict_function_approximations: dict=None, use_separate_model_for_target_V=True, update_period_model_for_target_V: int=100,
                 task=LearningTask.EPISODIC, criterion=LearningCriterion.DISCOUNTED, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0., func_adjust_alpha=None,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 debug=False):
        super().__init__(env,
                         dict_function_approximations=dict_function_approximations,
                         use_separate_model_for_target_V=use_separate_model_for_target_V,
                         update_period_model_for_target_V=update_period_model_for_target_V,
                         task=task, criterion=criterion,
                         alpha=alpha, gamma=gamma, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         store_history_over_all_episodes=True if task == LearningTask.CONTINUING else store_history_over_all_episodes)
        self.debug = debug

        # Ensure that the DISCOUNTED reward criterion has been passed for EPISODIC learning tasks
        # The reason for this check is that it is NOT possible to use TD to learn the AVERAGE reward value functions
        # (whose *state* value function is defined as V(x) = E[ sum(rewards until episode end) / T | X(0) = x ])
        # because TD learning relies on the Bellman equation and we cannot write the Bellman equation in this average reward
        # context because T is NOT known at the first transition step from t=0 to t=1,
        # which is how we build the Bellman equation, i.e. by splitting the value function definition into the first
        # reward observed when transition from t=0 to t=1 and the rest is re-branded as the value of the next state, V(X(1)).
        if task == LearningTask.EPISODIC and criterion == LearningCriterion.AVERAGE:
            # When the learning tasks are either EPISODIC or CONTINUING (which is the case at the moment of this writing (2023/12/25),
            # raising a value error for the above condition is equivalent to the following (more intuitive) condition that must be satisfied:
            # AVERAGE reward criterion => CONTINUING learning task
            raise ValueError("The EPISODIC learning task in TD learning requires the DISCOUNTED reward criterion, however the AVERAGE reward criterion was specified)")

        self.lmbda = lmbda
        # Eligibility traces for learning V
        self._z_V = np.zeros(self.V.getDimension())
        self._z_V_all = np.zeros((0, self.V.getDimension()))  # Historic information
        # Eligibility traces for learning Q
        if self.Q is not None:  # We may not want to learn Q(s,a) to save time (e.g. when learning policies based on the advantage function which only requires estimation of V(s))
            self._z_Q = np.zeros(self.Q.getDimension())
            self._z_Q_all = np.zeros((0, self.Q.getDimension()))  # Historic information
        # Eligibility traces for learning A, which are ALWAYS TABULAR (for the Generalized Advantage Estimation (GAE) --Ref: https://arxiv.org/abs/1707.06347, Schulman et al. (2017))
        self._z_A = np.zeros(self.env.getNumStates() * self.env.getNumActions())
        self._z_A_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions()))  # Historic information

        # (Nov-2020) Product of alpha and z (the eligibility trace)
        # which gives the EFFECTIVE alpha value of the Stochastic Approximation algorithm
        # Goal: Compare the rate of convergence of the non-adaptive vs. the adaptive TD(lambda)
        # by keeping track of the effective alpha as a FUNCTION of the episode number for EACH STATE
        # Each episode is a different row of the _alphas_effective array and the states are across the columns.
        self._times_nonzero_update = [[] for _ in self.env.getAllStates()]
        self._alphas_effective = np.zeros((0, self.V.getDimension()))

    def _reset_at_start_of_episode(self, reset_episode=True):
        """
        16-Jul-2025: Parameter reset_episode controls whether the eligibility traces are also reset under CONTINUING learning tasks.
        This reset should NOT happen under CONTINUING learning tasks so that states visited BEFORE
        the episode ended are also impacted by the rewards observed in the new episode.

        Note that we use a parameter name that is not so directly related to eligibility traces because
        information that affects the weight applied to each state and state-action in the eligibility traces,
        namely the learning rate alpha which is affected by the state and state-action counts,
        is controlled by the superclass learner which doesn't know about eligibility traces.
        """
        super()._reset_at_start_of_episode(reset_episode=reset_episode)
        if reset_episode or self.task == LearningTask.EPISODIC:
            self.reset_traces()

    def reset_traces(self):
        self._z_V[:] = 0.
        self._z_V_all = np.zeros((0, self.V.getDimension()))
        if self.Q is not None:  # We may not want to learn Q(s,a) to save time (e.g. when learning policies based on the advantage function which only requires estimation of V(s))
            self._z_Q[:] = 0.
            self._z_Q_all = np.zeros((0, self.Q.getDimension()))
        self._z_A[:] = 0.
        self._z_A_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions()))

        # The effective alphas correspond to the alpha learning rates multiplied by the eligibility traces, as that gives the actual update strength of the value functions
        # They are only computed for the learning of V, not of Q
        # (because this is stored for information purposes --e.g. plots of the eligibility traces to check if things are working properly)
        self._alphas_effective = np.zeros((0, self.V.getDimension()))

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda

    def learn(self, t, state, action, next_state, reward, done, info):
        if info.get('update_trajectory_and_average_reward', True):
            # We may not want to update the trajectory when using this call just to learn the value functions.
            # This is the case when using episodes under a continuing learning task context: in that case, the value functions
            # of the terminal state are normally learned at the start of the next episode (before updating the environment's state, which is a START state at that point,
            # and which is passed to this method as the value of next_state)
            # and thus the state of the environment and action taken should NOT be recorded because they have already
            # been recorded at the previous step, when the episode ended (a learning materialized here by the `done` block below)
            # --see also discrete.Simulator._run_single() and search for 'LearningTask.CONTINUING'
            self._update_trajectory(t, state, action, reward)  # This method belongs to the Learner super class defined in learners.episodic.discrete
            self._update_average_reward()
        self._update_visit_counts(t, state, action)

        # Add the state and next_state just visited to the known set of environment states which may be used when we need information about the size of the environment
        # (without using knowledge about the environment that the agent is not expected to know).
        # See the comment for the environment_set attribute in the GenericLearner class for a couple of use cases.
        super().updateKnownEnvironmentSet({state, next_state})

        # Whether to use TRUE GAE to learn the advantage function (as opposed to GAE)
        use_true_GAE = False

        # Compute the delta values used for the update of each value function
        # NOTE: We compute the delta separately, and NOT inside the functions that update the value functions,
        # because the delta information is needed by the adaptive TD(lambda) learner and implementing a specific
        # function that computes the delta values increases DRY implementation.
        delta_V, delta_Q, delta_V_fixed = self._compute_deltas(state, action, next_state, reward, info)

        # Update the eligibility trace
        self._updateZ(state, action, self.lmbda, delta_V=delta_V, delta_Q=delta_Q, use_true_GAE=use_true_GAE)

        if self.V.isTabular() or info.get('learning_mode', LearningMode.ONLINE) == LearningMode.ONLINE:
            # IMPORTANT: (2025/08/18) We do NOT update the target V(s) here because the value of `t` is NOT always the total number of simulation steps taken so far, as t may represent the time step within the episode.
            # Update target model if the period has been fulfilled
            # (We use `t+1` and NOT `t` because the first learning step has t = 0 and we do NOT want to update the target model at the very beginning)
            # Also, this allows the update of the model parameters at the end of an episode, since at that point t = -1 (see call to self.learn() in _run_single() and _run_single_continuing_task())
            #if not self.V.isTabular() and (t + 1) % self.update_period_model_for_target_V == 0:
            #    self.V_target.setModelParameters(self.V.getModelParameters())

            #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_over_all_episodes[state], self.getAlphaForState(self.env.getIndexFromState(state))))
            # Store the learning rates to be used in the value functions update
            self.store_learning_rate(self.getAlphasByState())

            # Update the action value functions
            # IMPORTANT: (2024/08/12) For the continuous state case that uses neural networks to approximate value functions,
            # we need to update Q first and then V o.w. we get the error that I do NOT understand:
            # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation?"
            # More info:
            # - https://github.com/pytorch/pytorch/issues/39141
            # - https://stackoverflow.com/questions/57631705/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modi
            self._updateQ(delta_Q, state=state, action=action)

            # Update the state value function
            # Retrieve the V(s) value BEFORE its update, in order to use it for the TRUE online TD(lambda) used by self._updateA_GAE()
            # to compute the Generalized Advantage Estimation (GAE)
            V_old = self.V.getValue(state)

            ###### TEMPORARY-TD(LAMBDA)
            self._updateV(delta_V, state=state)
            # (2025/10/20) When lambda > 0 and we learn V(s) using our own TD(lambda) learner (as opposed to the Adam optimizer)
            # (this option is currently hard-coded in this file --see sections `###### TEMPORARY-TD(LAMBDA)`)
            # we use GAE to learn the advantage function, therefore the TD error used in the update of V(s) is replaced by the current contribution to the GAE,
            # namely by `\tilde{H} := TD-error * "eligibility trace for H at current state and action"`,
            # where TD-error should be the one-step TD-error computed on a FIXED V(s) estimation (within an episode or epoch),
            # and this is why we use delta_V_fixed to multiply the eligibility trace self._z_A, as opposed to using `delta`.
            # (for more details, see my notes at the back of the SAS loose sheets, with date 15-Oct-2025 then copied to the SPSS notebook).
            #self._updateV(delta_V_fixed*self._z_A[self.A.getLinearIndex(self.env.getIndexFromState(state), action)] if self.lmbda > 0 else delta_V_fixed, state=state)
            ###### TEMPORARY-TD(LAMBDA)

            # From Sutton, page 300, where they talk about TRUE online TD(lambda)
            # This is an approximation of the actual difference in V(S(t)) before and after the update,
            # because rigorously we should use V_new_minus_old = V_t(S(t)) - V_{t-1}(S(t)) and here we are using V_new_minus_old = V_{t+1}(S(t)) - V_t(S(t)),
            # but it should be perfectly fine.
            # (The reason we use the approximation is that, using the correct difference requires storing a past value of V(.) which is not so straightforward)
            V_new_minus_old = self.V.getValue(state) - V_old

            # Update the advantage function
            self._updateA_GAE(delta_V_fixed, state=state, action=action, V_new_minus_old=V_new_minus_old if use_true_GAE else 0.0)
            #self._deprecated_updateA(state, action, delta_V)
            #self._updateA(delta_V, state=state, action=action)

            # We store the effective learning rates alpha
            # (effective in terms of  the eligibility trace that affects the delta values used when updating V above)
            if self.V.isTabular():
                # As many learning rates alpha as number of state-actions: each state-action affected by the eligibility trace will have their own alpha
                _alphas = self.getAlphasByState()
            else:
                # Use the alpha associated to the currently visited state as learning rate for ALL states visited in the past,
                # (i.e. `_alphas` is a scalar value) as the learning rate needs to multiply the eligibility trace vector _z_V
                # whose dimension is NOT the number of states in the environment, but the dimension of the theta vector parameterizing the value function.
                _alphas = self.getAlphaForState(self.env.getIndexFromState(state))
            self._alphas_effective = np.r_[self._alphas_effective, (_alphas * self._z_V).reshape(1, len(self._z_V))]
                ## NOTE: We need to reshape the product alpha*z because _alphas_effective is a 2D array with as many rows as
                ## the number of episodes run so far and as many columns as the number of states. The length of alpha*z
                ## is the number of states which should be laid out across the columns when appending a new row to
                ## _alphas_effective using np.r_[].

            # Update alpha for the next iteration for "by state counts" update
            #print("Learn: state = {}, next_state = {}, done = {}".format(state, next_state, done))
            if not self.adjust_alpha_by_episode and info.get('update_alphas', True):
                self._update_alphas(state, action)

        # Store the transition just observed (to be potentially used for OFFLINE or BATCH learning of value functions)
        # We also store also the eligibility traces (of all state-actions at the moment of the currently visited state-action)
        # so that we can learn the advantage function OFFLINE using TD(lambda) (as opposed to TD(0)) when lambda > 0.
        self.store_transition(info.get('transition_type', "MC"), t, state, action, next_state, reward, self._z_A)

        if done and info.get('update_trajectory_and_average_reward', True):
            # TEMPORARY-2025/01/14: The condition on 'update_trajectory_and_average_reward' was added today and is linked to the current implementation of the CONTINUING average reward
            # as an adjustment of the EPISODIC average reward, as explained in Learner.update_average_reward(). This new condition is only False when updating the value functions
            # at the end of a fictitious episode in CONTINUING tasks or when learning at the end of the simulation happening just after a reset of the environment due to reaching
            # a terminal state. In any of those two situations, we should NOT call the self.learn_at_episode_end() method here because the method updates the trajectory
            # that has already been updated by the methods calling this LeaTDLambda.learn() method: _run_single(), _run_single_continuing_task() and
            # run_exploration_and_learn_value_functions() in discrete.Simulator. See more details in those methods.
            # 2025/08/15: The condition defining parameter `update_counts` in self.learn_at_episode_end() is based on defining the conditions under which
            # this is the only opportunity to update the state and state-action visit counts at the end of an episode (i.e. when `done`, as is the case here,
            # which does NOT necessarily imply that a TERMINAL state has been reached --it can imply that the simulation ended), namely when EITHER:
            # - the learning task is EPISODIC.
            # - the learning task is CONTINUING and the maximum simulation time has been reached, i.e. this is the very last step of the simulation!
            # In either case, there will be NO next step of the simulation that will allow us to update the state and state-action visit counts,
            # in the first case, because by definition of EPISODIC learning task, when `done` there is NEVER a next step, either because the simulation ended
            # or because a terminal state has been reached; in the second case, because there will be no opportunity to update the state and state-action
            # visit counts when the process transitions away from this end state because the simulation ENDS here.
            # (We note that in the CONTINUING learning case, the visit counts of terminal states will be updated when the process transitions from the terminal
            # state to a start state (which may NOT be at the very next iteration step, for instance when the learner is a Fleming-Viot learner!),
            # at which moment LEARNING of the terminal state takes place and thus the alpha learning rates for the terminal state will be adjusted
            # by the UPDATED state and state-action visit counts. So we are in business by not updating the visit counts when the simulation does NOT end now.)
            # Recall again, that the end state can be either a terminal state (a TRUE "done episode" situation) or any other state (a TRUNCATED "done episode").
            # It's important to recall also that the visit counts have a direct impact on the state- and state-action-specific learning rates alpha,
            # which have a very important influence on the learning process!
            # Out of the above two situations, the only important context in which the update of the visit counts has a really impact on learning
            # is the CONTINUING learning task case, where terminal states do not necessarily value zero. For the EPISODIC learning task, updating
            # the end state visit counts has an impact on learning if the end state is NOT a terminal state, but such end state will normally have
            # large volatility (i.e. it will not always be the same), thus diluting the potential effect of not updating its visit counts at this time.
            self.learn_at_episode_end(t+1, next_state, update_counts=self.getLearningTask() == LearningTask.EPISODIC or info.get('max_time_steps_reached', False))

    def learn_at_episode_end(self, T, state_end, update_counts=True):
        """
        Performs the changes to the learner that must be done at the end of an episode, such as:
        - update of the average reward
        - trajectory storage
        - update of state count (of the end state)

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.

        state_end: int
            Index of the state at which the episode ends.

        update_counts: (opt) bool
            Whether the state visit count for the end state should be updated as well.
            We should set this to False for CONTINUING learning task when this is NOT the last step of the simulation
            (i.e. when the maximum number of simulation steps has NOT been reached), because if this is NOT the last step of the simulation,
            the visit count for the current end state (which must be a terminal state, by definition of episode when the maximum number of steps
            has not been reached) will be updated when learning the terminal state, i.e. when transitioning from the terminal state to a start state.
            We also might want to set this to False to avoid a too aggressive decrease of the learning rate alpha that in the end prevents learning,
            especially when a state is visited but no reward innovation is observed, meaning that the learning step is sort of "useless" because
            the delta(V) value to apply to the new V(s) is 0.
        """
        # Update the average reward over ALL episodes (needed for the AVERAGE reward criterion scenario)
        self.update_average_reward(T, state_end)

        if self.debug: #and self.episode > 45: # Use the condition on `episode` in order to plot just the last episodes
            self._plotZ()
            self._plotAlphasEffective()

        self.store_trajectory_at_episode_end(T, state_end, debug=self.debug)
        if update_counts:
            self._update_visit_counts(T, state_end, np.nan)

        # Update alpha for the next iteration for "by episode" updates
        if self.adjust_alpha_by_episode:
            for s in range(self.env.getNumStates()):
                self._update_alphas(s, np.nan)

    def _compute_deltas(self, state, action, next_state, reward, info):
        """
        Computes the delta values to be used for the state value and action value functions update

        The delta value of the action value function is computed using the Expected SARSA approach,
        i.e. the expected Q-value over all possible next actions is used. This is theoretically equal to
        the state value function of the next state and thus its current estimate is used.

        Note that, since we use an *estimate* of the state value function V, as opposed to the true state value function,
        the result will most likely NOT be the same as computing the expected value of the estimated Q-values over all actions,
        as such computation does not necessarily produce a value that coincides with the estimated state value function.
        """
        if self.useSeparateModelForTargetV():
            assert not self.V.isTabular(), "Using a separate TARGET model for V(s) is only possible when V(s) is NOT tab"
            # Value of the next state
            V_target_value = reward + self.gamma * self.V_target.getValue(next_state) #- np.mean(self.V_target.getValues())

            # Value of the current state
            V_value = self.V.getValue(state) #- np.mean(self.V.getValues())
            # (2025/10/07) Use the following if we want to use a FIXED delta value that is NOT updated as the state value function is updated during the ONLINE learning process
            # This should perhaps give more stability to the learning process, and actually is the delta value that should be used to compute the GAE of the advantage function,
            # where the V function  used in the computation of ALL delta values should be always the same (e.g. see Schulman et al. (2015), who introduced GAE).
            # (2026/05/16) HOWEVER, when tried today (running the V(s) estimation process from value_functions.py on a 10x14 2D labyrinth, BOTH with TD(lambda) and FV(lambda))
            # the estimated V(s) DIVERGES!!! WHY??? (Need to investigate further)
            #V_value = self.V_target.getValue(state)

            # delta(V) computed on a FIXED V(s) function, as fixed as the FIX level provided by the target model for V(s)
            # Goal: Use it in the estimation of the advantage function which is supposed to use a TD error (delta) computed on a fixed estimate V(s)
            # (see my SPSS notebook, entry on 09-Oct-2025)
            delta_V_fixed = reward + self.gamma * self.V_target.getValue(next_state) - self.V_target.getValue(state)
        else:
            # Tabular case and NN case with NO target V(s) model
            V_target_value = reward + self.gamma * self.V.getValue(next_state)
            V_value = self.V.getValue(state)
            delta_V_fixed = V_target_value - V_value
        delta_V = V_target_value - V_value
        if self.Q is not None:  # We may not want to learn Q(s,a) to save time (e.g. when learning policies based on the advantage function which only requires estimation of V(s))
            # Note that for the Q value of the next state and next action we use its expected value
            # (over all possible actions) as it is done by the Expected SARSA learning of the Q function.
            # This avoids having to choose a particular next action for which we would require a new parameter
            # such as the epsilon value of the epsilon-greedy next action strategy.
            delta_Q = reward + self.gamma * self._expected_next_Q(next_state) - self.Q.getValue(state, action)
            # Use the following if we want to choose the action with the highest Q-value at the next state
            #delta_Q = reward + self.gamma * self._max_next_Q(next_state) - self.Q.getValue(state, action)
        else:
            delta_Q = 0.0

        # Check whether we are learning the differential value function
        # (average reward criterion for the continuing learning task context) and adjust delta accordingly
        # Ref: Sutton, pag. 250
        if self.criterion == LearningCriterion.AVERAGE:
            # Recall that in the LeaTD constructor we have checked that AVERAGE reward criterion => CONTINUING learning task
            # (the converse however is not necessarily true, as a continuing learning task may still use a discount factor;
            # if this is the case, no average reward correction is applied to the value functions, as the discount prevents the sum of rewards from diverging;
            # note however that Sutton on pag. 251/252 defines a "discounted" state value function where still an average reward correction is present in its definition;
            # however, in that case the discount factor gamma -> 1, and this is exclusively done to make the differential return be well defined for non-ergodic Markov chains)
            if info.get('average_reward') is not None:
                average_reward_correction = info.get('average_reward')
            else:
                # Note: The average reward that is used for the correction of the value functions to obtain the differential value functions
                # should be the average reward observed over ALL episodes because such average reward is associated to a CONTINUING learning task
                # which sees no episodes... i.e. if the simulation is performed in episodes, it is merely an IMPLEMENTATION detail, NOT a LEARNING characteristic.
                # This is why here we call the GenericLearner.getAverageReward() method which retrieves the average reward observed over the whole simulation,
                # regardless of any implementation-related episodes.
                average_reward_correction = self.getAverageReward()
            #print(f"[_compute_deltas()] Value functions corrected by average reward = {average_reward_correction:.4g}")
            delta_V -= average_reward_correction
            delta_Q -= average_reward_correction
            delta_V_fixed -= average_reward_correction

        return delta_V, delta_Q, delta_V_fixed

    def _updateZ(self, state, action, lmbda, delta_V=None, delta_Q=None, use_true_GAE=False):
        """
        Updates the eligibility traces used for learning V and those used for learning Q

        Parameter `use_true_GAE` enables changing the calculation of the eligibility trace for the advantage function (self._z_A)
        in order to apply TRUE GAE when learning the advantage function with _updateA_GAE().
        """
        ###### TEMPORARY-TD(LAMBDA)
        # Note: delta_V is NOT used when V(s) is TABULAR (see the self.V.getGradient() method of the tabular case
        gradient_V = self.V.getGradient(state, delta_V, is_learner_td_lambda=False)
        # Use the following condition `lmbda > 0` when we want to compute the gradient contributing to the eligibility trace as the gradient of V(s) w.r.t. the NN model parameters (as opposed to being computed from the gradient of the loss) --althogh I am not sure if there is a difference... (16-May-2026)
        #gradient_V = self.V.getGradient(state, delta_V, is_learner_td_lambda=lmbda > 0)
        ###### TEMPORARY-TD(LAMBDA)

        if gradient_V is not None:
            self._z_V = self.gamma * lmbda * self._z_V + \
                        gradient_V                                    # For every-visit TD(lambda)
                        #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
            self._z_V_all = np.r_[self._z_V_all, self._z_V.reshape(1, len(self._z_V))]

        if self.Q is not None:  # We may not want to learn Q(s,a) to save time (e.g. when learning policies based on the advantage function which only requires estimation of V(s))
            # Note: delta_Q is NOT used when Q(s,a) is TABULAR (see the self.Q.getGradient() method of the tabular case
            gradient_Q = self.Q.getGradient(state, action, delta_Q, is_learner_td_lambda=False) #lmbda > 0)
            if gradient_Q is not None:
                self._z_Q = self.gamma * lmbda * self._z_Q + \
                            gradient_Q
                self._z_Q_all = np.r_[self._z_Q_all, self._z_Q.reshape(1, len(self._z_Q))]

        # Eligibility traces for GAE, the update of the advantage function using the Generalized Advantage Estimation which allows implementing TD(lambda)
        # Actually GAE is the same as TD(lambda), but it just has a different name because what is updated is not the state value function, as in TD(lambda), but the advantage.
        A_vector = np.zeros(self.env.getNumStates() * self.env.getNumActions(), dtype=float)
        # Set the component of A_vector that will be affected (in _updateA()) by delta_V, to update the advantage of the currently visited state-action
        A_vector[self.A.getLinearIndex(self.env.getIndexFromState(state), action)] = 1.0
        # We define the eligibility trace of the advantage function following Sutton, page 300, where they talk about TRUE online TD(lambda).
        # The goal is to better implement GAE(lambda) (the Generalized Advantage Estimator) compared to plain GAE(lambda).
        # What we do here mimics TRUE online TD(lambda), which is based on defining the TD error as R(t+1) + gamma * V_t(S(t+1)) - V_{t-1}(S(t)),
        # i.e. by using the PREVIOUS estimate of v(S(t)) as subtracting predicted value, instead of the current estimate V_t(S(t)).
        # Ref: http://incompleteideas.net/book/first/ebook/node76.html
        self._z_A = self.gamma * lmbda * self._z_A + \
                    (1 - int(use_true_GAE) * self.gamma * lmbda * self._z_A[self.A.getLinearIndex(self.env.getIndexFromState(state), action)]) * A_vector
        self._z_A_all = np.r_[self._z_A_all, self._z_A.reshape(1, len(self._z_A))]

    def _updateV(self, delta, state):
        if delta != 0.0:
            if self.V.isTabular():
                # As many learning rates alpha as number of states: each state affected by the eligibility trace will have their own alpha

                # IMPORTANT: (2020/11/11) Note that we use self.getAlphasByState() and NOT self.getAlphaForState(state) to retrieve the alpha values for each state
                # as the former method gives the alpha value for EACH state in the eligibility trace, which may be different from the alpha value for the CURRENTLY visited `state`,
                # which is retrieved by the latter method.
                # In the latter case, we would be using the SAME learning rate alpha for the update of ALL states, and this is NOT how the alpha value should be applied.
                # (as we should apply the alpha associated to the state that decreases with the number of visits to EACH state --which happens differently).
                # However, using the same alpha seems to give slightly faster convergence than the state-based alpha strategy, at least in the gridworld environment.
                _alphas = self.getAlphasByState()
            else:
                # Use the alpha associated to the currently visited state as learning rate for ALL states visited in the past,
                # (i.e. `_alphas` is a scalar value) as the learning rate needs to multiply the eligibility trace vector _z_V
                # whose dimension is NOT the number of states in the environment, but the dimension of the theta vector parameterizing V(s).
                _alphas = self.getAlphaForState(self.env.getIndexFromState(state))

            #-- TESTING THE LEARNING PROCESS BY A NEURAL NETWORK BY PROVIDING THE TRUE FUNCTION VALUE
            # CONCLUSION: Option 2 works as long as we convert the sum we do in my V._compute_loss() function to a tensor!!!! (o.w. the gradient is zero! ARRRGHRHHRHHH!!!)
            #import torch
            #
            # 1) Computing the loss directly, giving the target and predicted values, which are the arguments of the loss
            #loss = self.V.loss(torch.tensor(self.env.getV()[state]), self.V._getValue(state))

            # 2) Computing the loss indirectly, by giving the state and the delta value observed (using my V._compute_loss() function to this end)
            #delta = self.env.getV()[state] - self.V.getValue(state)
            #loss = self.V._compute_loss(state, delta)

            #self.V.optimizer.zero_grad()
            #loss.backward()
            #self.V.optimizer.step()
            #-- TESTING THE LEARNING PROCESS BY A NEURAL NETWORK BY PROVIDING THE TRUE FUNCTION VALUE

            ###### TEMPORARY-TD(LAMBDA)
            self.V.updateWeights(state, delta, multiplier_delta=_alphas * self._z_V, is_learner_td_lambda=False)
            # Use the following condition `self.lmbda > 0` when we want to update the weights using the TD(lambda) gradient (defined as delta * multiplier_delta), as opposed to the gradient of the model loss
            #self.V.updateWeights(state, delta, multiplier_delta=_alphas * self._z_V, is_learner_td_lambda=self.lmbda > 0)
            ###### TEMPORARY-TD(LAMBDA)

    def _updateQ(self, delta, state, action):
        if delta != 0.0 and self.Q is not None:
            # DM-2025/06/22: We now use the alpha by state-action instead of the alpha by state as alpha for the update of Q,
            # which better takes into account the number of visits to each state AND action, not only to each state.

            if self.Q.isTabular():
                # As many learning rates alpha as number of state-actions: each state-action affected by the eligibility trace will have their own alpha

                # Reorganize the SxA array into an S*A 1D array grouped by state, i.e. all actions for the first state, then all actions for the second state, etc.
                # which is how the linearized Q values are organized.
                # Note that, if we just wanted to use the same alpha for all actions, we could use a np.repeat() call as follows:
                #   _alphas = np.repeat(self.getAlphasByState(), self.env.getNumActions())
                # Note that the above repeats each value making a layout of the alpha values as we need them, namely respecting the state-action layout in the feature matrix for Q,
                # grouped by state. Ex: if alphas = [2.5, 4.1, 3.0] for three different states, the repeat by 2 actions generates [2.5, 2.5, 4.1, 4.1, 3.0, 3.0]
                # i.e. the same alpha for all actions associated to the same state (which is what is needed, i.e. alphas on different actions grouped by state).
                _alphas2 = self.getAlphasByStateAction().reshape(-1)
            else:
                # Use the alpha associated to the currently visited state and action as learning rate for ALL state-actions visited in the past,
                # (i.e. `_alphas2` is a scalar value) as the learning rate needs to multiply the eligibility trace vector _z_Q
                # whose dimension is NOT the number of states in the environment, but the dimension of the theta vector parameterizing the value function.
                _alphas2 = self.getAlphaForStateAction(self.env.getIndexFromState(state), action)
            ###### TEMPORARY-TD(LAMBDA)
            self.Q.updateWeights(state, action, delta, multiplier_delta=_alphas2 * self._z_Q, is_learner_td_lambda=False)
            # Use the following condition `self.lmbda > 0` when we want to update the weights using the TD(lambda) gradient (defined as delta * multiplier_delta), as opposed to the gradient of the model loss
            #self.Q.updateWeights(state, action, delta, multiplier_delta=_alphas2 * self._z_Q, is_learner_td_lambda=self.lmbda > 0)
            ###### TEMPORARY-TD(LAMBDA)

    def _expected_next_Q(self, next_state):
        """
        Computes the expected Q value for the next state over all possible next actions (which is what is done by the Expected SARSA learner of the Q function)

        In the tabular context, when both V and Q are tabular, we use V(s) as an estimate of the expected Q value.
        Otherwise, we compute the PLAIN average of the Q values.
        Note that we do NOT compute the policy-weighted average, because this learner object does NOT have access to the current policy.
        If we want to compute the policy-weighted average, i.e. the ACTUAL expected Q value over all possible next actions,
        we should pass the current policy to the constructor of this class.
        """
        if self.V.isTabular() and self.Q.isTabular():
            # In the tabular context, we consider V(s) to be a fairly good estimate of the expectation of Q(s,A) over all actions A
            return self.V.getValue(next_state)
        else:
            # This means that V and Q are estimated with two different models (e.g. tabular and NN or two NN models)
            # => To avoid having a very large error when using V(s) as an estimate of Expected vale of Q(s,A) over all A, we compute the plain average of Q(s,A)
            # (note that we do NOT compute policy-weighted average because this object does not know about the policy.
            Q_mean = 0.0
            for action in range(self.env.getNumActions()):
                Q_mean += self.Q.getValue(next_state, action)
            Q_mean /= self.env.getNumActions()
            return Q_mean

    def _max_next_Q(self, next_state):
        """
        Computes the maximum Q value for the next state over all possible next actions (which is what is done by the Q-learning algorithm)
        """
        Q_max = -np.Inf
        for action in range(self.env.getNumActions()):
            Q_max = max(Q_max, self.Q.getValue(next_state, action))
        return Q_max

    def _updateA(self, delta, state, action):
        if delta != 0.0:
            # For details about the computation of _alphas2, see comments in the _updateQ() method
            if self.A.isTabular():
                # As many learning rates alpha as number of state-actions: each state-action affected by the eligibility trace will have their own alpha
                _alphas2 = self.getAlphasByStateAction().reshape(-1)
            else:
                # Use the alpha associated to the currently visited state and action as learning rate for ALL state-actions visited in the past,
                # (i.e. `_alphas2` is a scalar value) as the learning rate needs to multiply the eligibility trace vector _z_Q
                # whose dimension is NOT the number of states in the environment, but the dimension of the theta vector parameterizing the value function.
                _alphas2 = self.getAlphaForStateAction(self.env.getIndexFromState(state), action)
            ###### TEMPORARY-TD(LAMBDA)
            # Use the following condition `self.lmbda > 0` when we want to update the weights using the TD(lambda) gradient (defined as delta * multiplier_delta), as opposed to the gradient of the model loss
            self.A.updateWeights(state, action, delta, multiplier_delta=_alphas2 * self._z_A, is_learner_td_lambda=False)
            #self.A.updateWeights(state, action, delta, multiplier_delta=_alphas2 * self._z_A, is_learner_td_lambda=self.lmbda > 0)
            ###### TEMPORARY-TD(LAMBDA)

    def _updateA_GAE(self, delta, state, action, V_new_minus_old=0.0):
        """
        Updates the advantage function (when `delta` is not zero) following the Generalized Advantage Estimation (GAE)

        Note that the GAE is the adaptation of TD(lambda) to the estimation of the advantage function (as opposed to learning the state value function).
        The principle is the same, but the difference is that in the update formula for the advantage function there is NO learning rate alpha
        because the update is an exponentially-weighted combination of one-step TD errors (see the Schulman reference below).
        The absence of the learning rate alpha is clearly seen by considering the TD(0) case, where the advantage function is simply equal to the TD error,
        i.e. there is NO update as the one done for V(s), where `V(s) <- V(s) + alpha*delta`... here it's simply `H(s,a) <- delta`.

        Use the a `V_new_minus_old` value different from zero in order to use *TRUE* GAE (as described in Sutton, Chapter 12.5).

        Ref:
        Schulman et al. (2017), https://arxiv.org/abs/1707.06347 and better https://arxiv.org/pdf/1506.02438 --> for GAE
        Sutton (2018), Chapter 12.5, "True online TD(lambda)", pag. 300 --> for the TRUE online TD(lambda) which is adapted to GAE
        """
        if delta != 0.0:
            assert self.A.isTabular(), "The advantage function MUST be tabular, because it is estimated as the delta(V) error, " \
                                       "and THIS delta(V) value is the one that could be computed using a function approximation for V(s)"

            # Dummy vector signalling the currently visited state-action which defines the additional term being subtracted below when V_new_minus_old != 0.0
            A_vector = np.zeros(self.env.getNumStates() * self.env.getNumActions(), dtype=float)
            A_vector[self.A.getLinearIndex(self.env.getIndexFromState(state), action)] = 1.0
            self.A.setWeights(self.A.getWeights() + (delta + V_new_minus_old) * self._z_A - \
                                                    V_new_minus_old * A_vector)

    # DM-2025/06/20: Deprecated this method because it is only valid for TD(0)
    # as it only updates the advantage of the current state and action and not of the past states and actions visited during the trajectory --which are also affected by TD(lambda)!
    def _deprecated_updateA(self, state, action, advantage):
        """
        Sets the value of the Advantage function to the given value for the given state and action.
        An unbiased estimation of the advantage is the delta(V) observed when taking the given action at the given state, i.e. the TD error.
        """
        if self.env.isStateContinuous():
            # IMPORTANT: The advantage function is assumed to be TABULAR
            self.A.setValue(self.env.getIndexFromState(state), action, advantage)
        else:
            # Recall that _setWeight() assumes that the features are dummy features (so, at some point this would need to be updated)
            self.A._setWeight(state, action, advantage)

    def _plotZ(self):
        states2plot = self._choose_states2plot()
        plt.figure()
        plt.plot(self._z_V_all[:,states2plot], '.-')
        plt.legend(states2plot)
        ax = plt.gca()
        ax.set_xlabel("time step")
        ax.set_ylabel("z")
        start_state = self._states[0] if len(self._states) > 0 else None
        ax.set_title("Eligibility trace by time step (Episode {} - start state = {})".format(self.episode, start_state))
        plt.pause(0.001)
        plt.show()

    def _plotAlphasEffective(self):
        states2plot = self._choose_states2plot()
        plt.figure()
        #for s in states2plot:
        #   plt.plot(self._times_nonzero_update[s], self._alphas_effective[s], '.-')
        #plt.plot(self._times_nonzero_update[self.env.getNumStates()-1], self._alphas_effective[self.env.getNumStates()-1], '.-')
        plt.plot(self._alphas_effective[:, states2plot], '.-')
        plt.legend(states2plot)
        ax = plt.gca()
        #ax.set_xlim([0,len(self._states)-1])
        ax.set_ylim((0, 1))
        ax.set_xlabel("time step")
        ax.set_ylabel("alpha*z")
        start_state = self._states[0] if len(self._states) > 0 else None
        ax.set_title("Effective learning rate (alpha*z) by time step (Episode {} - start state = {})".format(self.episode, start_state))
        plt.pause(0.001)
        plt.show()

    def _choose_states2plot(self):
        from Python.lib.environments.gridworlds import EnvGridworld1D
        from Python.lib.environments.mountaincars import MountainCarDiscrete
        if issubclass(self.env.__class__, EnvGridworld1D) and self.env.getNumStates() > 12:
            states2plot = list(range(9, 12))
        elif isinstance(self.env, MountainCarDiscrete):
            # Assuming MountainCar environment
            states2plot = list(range(100, 104))
        else:
            # Choose the states around the middle state
            states2plot = list(range(int(self.env.getNumStates()/2)-1, int(self.env.getNumStates()/2)+2))

        return states2plot

    def getElibilityTraceForStateValue(self):
        return self._z_V

    def getElibilityTraceForActionValue(self):
        return self._z_Q

    def getElibilityTraceForAdvantage(self):
        return self._z_A


class LeaTDLambdaAdaptive(LeaTDLambda):
    
    def __init__(self, env,
                 dict_function_approximations=None,
                 use_separate_model_for_target_V=True, update_period_model_for_target_V=100,
                 task=LearningTask.EPISODIC, criterion=LearningCriterion.DISCOUNTED, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0., func_adjust_alpha=None,
                 lambda_min=0., lambda_max=0.99, adaptive_type=AdaptiveLambdaType.ATD,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 burnin=False, plotwhat="boxplots", fontsize=15, debug=False):
        super().__init__(env, dict_function_approximations=dict_function_approximations, use_separate_model_for_target_V=use_separate_model_for_target_V, update_period_model_for_target_V=update_period_model_for_target_V,
                         task=task, criterion=criterion, alpha=alpha, gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         store_history_over_all_episodes=True if task == LearningTask.CONTINUING else store_history_over_all_episodes,
                         debug=debug)

        # List that keeps the history of ALL lambdas used at EVERY TIME STEP
        # (i.e. all states are mixed up here and if we want to identify which state the lambda corresponds to
        # we need to look at the history of states provided by the learner)
        self._lambdas = []
        # Minimum lambda for the adaptive lambda so that there is still some impact
        # in past states at the beginning when all state values are equal and equal to 0
        self.lambda_min = lambda_min
        # Maximum lambda for the adaptive lambda which guarantees convergence
        self.lambda_max = lambda_max
        # Type of adaptive lambda (FULLY adaptive or HOMOGENEOUS)
        self.adaptive_type = adaptive_type
        # Whether to perform a burn-in learning using a constant lambda
        # at the beginning to accelerate learning
        self.burnin = burnin
        # Type of plot to generate to analyze the adaptive lambdas by state
        self.plotwhat = plotwhat
        # Reference font size to use in 2D image plot of the lambdas by state
        # to show the number of cases and/or the lambda value
        self.fontsize = fontsize

        # Counter of state visits WITHOUT resetting the count after each episode
        # (This MIGHT be used to decide whether we should use the adaptive or non-adaptive lambda
        # based on whether the delta information from which the agent learns already contains
        # bootstrapping information about the value function at the next state)
        self.state_counts_noreset = np.zeros(self.env.getNumStates())

        #-- Variables used in the HOMOGENEOUS adaptive type case
        self._gradient_V_all = np.zeros((0, self.env.getNumStates()))

        #-- Variables for lambda statistics over all episodes
        # List of lists to store the lambdas used for each state in each episode
        self._lambdas_in_episode = [[] for _ in self.env.getAllStates()]
        # Store all the lambdas over all episodes
        # This is a 3D list indexed by:
        # - episode number
        # - state
        # - visit to the state in the episode
        self._all_lambdas_by_episode = []
        # For count, mean and std by state
        self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
        self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
        self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

        # Keep track of the average lambda by episode
        # computed over the lambdas observed during the TRAJECTORY
        # (i.e. it's not an average over the lambdas by state!)
        self.lambda_mean_by_episode = []

    def reset(self, reset_episode=False, reset_value_functions=False, reset_average_reward=False):
        super().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions, reset_average_reward=reset_average_reward)
        if reset_episode:
            self.lambda_mean_by_episode = []
            self._all_lambdas_by_episode = []
            del self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2
            self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
            self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
            self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

    def _reset_at_start_of_episode(self, reset_episode=True):
        """
        16-Jul-2025: Parameter reset_episode controls whether the eligibility traces are also reset under CONTINUING learning tasks.
        This reset should NOT happen under CONTINUING learning tasks so that states visited BEFORE
        the episode ended are also impacted by the rewards observed in the new episode.

        Note that we use a parameter name that is not so directly related to eligibility traces because
        information that affects the weight applied to each state and state-action in the eligibility traces,
        namely the learning rate alpha which is affected by the state and state-action counts,
        is controlled by the superclass learner which doesn't know about eligibility traces.
        """
        super()._reset_at_start_of_episode(reset_episode=reset_episode)
        if reset_episode or self.task == LearningTask.EPISODIC:
            self.reset_traces()

    def reset_traces(self):
        super().reset_traces()  # This is NEW as of 14-Jul-2025... I don't understand why the eligibility traces were not reset before... (i.e. _z_V, _z_V_all, _z_Q, _z_Q_all)
                                # Well, at least we should reset _z_V_all and _z_Q_all because it is true that _z_V and _z_Q are computed from _gradient_V_all and _gradient_Q_all
                                # (see the _updateZ() method below).
        self._gradient_V_all = np.zeros((0, self.env.getNumStates()))
        if self.Q is not None:
            self._gradient_Q_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions()))
        self._lambdas = []
        self._lambdas_in_episode = [[] for _ in self.env.getAllStates()]

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None,
                  lambda_min=None, lambda_max=None, adaptive_type=None,
                  burnin=False):
        super().setParams(alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.lambda_min = lambda_min if lambda_min is not None else self.lambda_min
        self.lambda_max = lambda_max if lambda_max is not None else self.lambda_max
        self.adaptive_type = adaptive_type if adaptive_type is not None else self.adaptive_type
        self.burnin = burnin if burnin is not None else self.burnin

    def learn(self, t, state, action, next_state, reward, done, info):
        if info.get('update_trajectory_and_average_reward', True):
            # We may not want to update the trajectory when using this call just to learn the value functions.
            # See the comment in the learn() method of the super class (normally LeaTDLambda) for an use case.
            self._update_trajectory(t, state, action, reward)  # This method belongs to the Learner super class defined in learners.episodic.discrete
            self._update_average_reward()
        self._update_visit_counts(t, state, action)

        # See comment about this step in LeaTDLambda.learn()
        super().updateKnownEnvironmentSet({state, next_state})

        # See comment in the constructor of the meaning of this attribute, which is exclusively used in the adaptive lambda learner
        self.state_counts_noreset[state] += 1

        delta_V, delta_Q, delta_V_fixed = self._compute_deltas(state, action, next_state, reward, info)

        # Decide whether we do adaptive or non-adaptive lambda at this point
        # (depending on whether there is bootstrap information available or not)
        # TODO: (2020/04/21) Adapt the check on whether a value function has been modified at least once that works also for the case when the initial state value function is not 0 (e.g. it is random).
        # (2023/11/12) The implementation of the above TO-DO would be in the line of using the state_counts_noreset attribute, which for some reason is now commented out.
        delta = delta_V
        if not done and self.burnin and self.V.getValue(next_state) == 0: # self.state_counts_noreset[next_state] == 0:
            # The next state is non terminal and there is still no bootstrap information
            # about the state value function coming from the next state
            # => Do NOT do an adaptive lambda yet... so that some learning still happens.
            # In fact, the adaptive lambda may suggest NO learning due to the absence of innovation
            # in case the reward of the current step is 0 (e.g. in gridworlds receiving reward only at terminal states)
            #print(f"*** ENTERED BURNIN (reward={reward})")
            lambda_adaptive = self.lmbda
            #print("episode: {}, t: {}, lambda (fixed) = {}".format(self.episode, t, lambda_adaptive))
            #print("episode: {}, t: {}, next_state: {}, state_counts[{}] = {} \n\tstate counts: {}\n\tlambda(adap)={}" \
            #      .format(self.episode, t, next_state, next_state, self.state_counts_noreset[next_state], self.state_counts_noreset, lambda_adaptive))
        else:
            #-- Adaptive lambda
            # Define the relative target error delta by dividing the bootstrap delta (for now only for V, not for Q) to a reference value defined below
            #ref_value = self.V.getValue(state)                                         # reference value is the value of the current state
            if self.task == LearningTask.EPISODIC:
                # Reference value is the average value over NON-TERMINAL states (whose value is always 0, so they should no te included in the average)
                ref_value = np.mean( np.abs(self.V.getValues()[self.env.getNonTerminalStates()]) )
            else:
                # Reference value is the average value over all states
                ref_value = np.mean(np.abs(self.V.getValues()))

            delta_relative = delta / ref_value if ref_value != 0 \
                                               else 0. if delta == 0. \
                                               else np.Inf
            # Relative delta that prevents division by 0 (i.e. delta_relative = exp(|delta|) / exp(|value|))
            #delta_relative = np.exp( np.abs(delta) - np.abs(self.V.getValue(state)) )

            # Compute lambda as a function of delta or relative delta
            lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta_relative) ), self.lambda_max )
            #lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta) ), self.lambda_max )

            if self.debug:
                print("episode: {}, t: {}: TD ERROR != 0 => delta = {}, delta_rel = {}, lambda (adaptive) = {}\n" \
                        "trajectory so far: {}" \
                        "--> V(s): {}".format(self.episode, t, delta, delta_relative, lambda_adaptive, self._states, self.V.getValues()))

        # Store the learning rates to be used in the value functions update
        self.store_learning_rate(self.getAlphasByState())
        # Update the eligibility trace
        self._updateZ(state, action, lambda_adaptive)
        # Update the action value functions
        # IMPORTANT: (2024/08/12) For the continuous state case that uses neural networks to approximate value functions,
        # we need to update Q first and then V o.w. we get the error that I do NOT understand:
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation?"
        # More info:
        # - https://github.com/pytorch/pytorch/issues/39141
        # - https://stackoverflow.com/questions/57631705/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modi
        self._updateQ(delta_Q, state=state, action=action)

        # Update the state value function
        # Retrieve the V(s) value BEFORE its update, in order to use it for the TRUE online TD(lambda) used by self._updateA_GAE()
        # to compute the Generalized Advantage Estimation (GAE)
        V_old = self.V.getValue(state)
        self._updateV(delta_V, state=state)
        # From Sutton, page 300, where they talk about TRUE online TD(lambda)
        # This is an approximation of the actual difference in V(S(t)) before and after the update,
        # because rigorously we should use V_new_minus_old = V_t(S(t)) - V_{t-1}(S(t)) and here we are using V_new_minus_old = V_{t+1}(S(t)) - V_t(S(t)),
        # but it should be perfectly fine.
        V_new_minus_old = self.V.getValue(state) - V_old

        # Update the advantage function
        self._updateA_GAE(delta_V_fixed, state=state, action=action, V_new_minus_old=V_new_minus_old)
        #self._deprecated_updateA(state, action, delta_V)
        #self._updateA(delta_V, state=state, action=action)

        # The effective alphas are only computed for the learning of V, not of Q
        # (as this is only stored for information purposes --e.g. plots of the eligibility traces to check if things are working properly)
        if self.V.isTabular():
            # As many learning rates alpha as number of state-actions: each state-action affected by the eligibility trace will have their own alpha
            _alphas = self.getAlphasByState()
        else:
            # Use the alpha associated to the currently visited state as learning rate for ALL states visited in the past,
            # (i.e. `_alphas` is a scalar value) as the learning rate needs to multiply the eligibility trace vector _z_V
            # whose dimension is NOT the number of states in the environment, but the dimension of the theta vector parameterizing V(s).
            _alphas = self.getAlphaForState(self.env.getIndexFromState(state))
        self._alphas_effective = np.r_[self._alphas_effective, (_alphas * self._z_V).reshape(1, len(self._z_V))]
            ## NOTE: We need to reshape the product alpha*z because _alphas_effective is a 2D array with as many rows as
            ## the number of episodes run so far and as many columns as the number of states. The length of alpha*z
            ## is the number of states which should be laid out across the columns when appending a new row to
            ## _alphas_effective using np.r_[].

        # Keep history of used lambdas
        self._lambdas += [lambda_adaptive]
        # Update the history of lambdas used
        self._lambdas_in_episode[state] += [lambda_adaptive]
        self._all_lambdas_n[state] += 1
        self._all_lambdas_sum[state] += lambda_adaptive
        self._all_lambdas_sum2[state] += lambda_adaptive**2

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode and info.get('update_alphas', True):
            self._update_alphas(state, action)

        # Store the transition just observed (to be potentially used for OFFLINE or BATCH learning of value functions)
        self.store_transition(info.get('transition_type', "MC"), t, state, action, next_state, reward, self._z_A)

        if done and info.get('update_trajectory_and_average_reward', True):
            # TEMPORARY-2025/01/14: The condition on 'update_trajectory_and_average_reward' was added today and is linked to the current implementation of the CONTINUING average reward.
            # For more details about this and about the definition of parameter `update_counts=` below, see the comment I wrote in LeaTDLambda.learn() of the super class.
            self.learn_at_episode_end(t+1, next_state, update_counts=self.getLearningTask() == LearningTask.EPISODIC or self.getLearningTask() == LearningTask.CONTINUING and info.get('max_time_steps_reached', False))
            self._store_lambdas_in_episode()

        if self.debug:
            print("t: {}, delta = {:.3g} --> lambda = {:.3g}".format(t, delta, lambda_adaptive))
            print("\tV(s={}->{}) = {}".format(state, next_state, self.V.getValue(state)))
            if done:
                import pandas as pd
                pd.options.display.float_format = '{:,.2f}'.format
                print(pd.DataFrame( np.c_[self._z_V, self.V.getValues()].T, index=['_z', 'V'] ))
    
            #input("Press Enter...")

    def _updateZ(self, state, action, lmbda):
        if self.debug and False:
            print("")
            print("state = {}: lambda = {:.2f}".format(state, lmbda))
        if self.adaptive_type == AdaptiveLambdaType.ATD:
            super()._updateZ(state, action, lmbda)
        else:
            # In the HOMOGENEOUS adaptive lambda we need to store the HISTORY of the gradient
            # (because we need to retroactively apply the newly computed lambda to previous eligibility traces)
            gradient_V = self.V.X[:, state]      # Note: this is returned as a ROW vector, even when we retrieve the `state` COLUMN of matrix X
            gradient_Q = self.Q.X[:, self.Q.getLinearIndex(self.env.getIndexFromState(state), action)] if self.Q is not None else None
            # Use the following calculation of the gradient for FIRST-VISIT TD(lambda)
            # (i.e. the gradient is set to 0 if the current visit of `state` is not the first one)
            #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
            self._gradient_V_all = np.r_[self._gradient_V_all, gradient_V.reshape(1, len(gradient_V))]
            if gradient_Q is not None:
                self._gradient_Q_all = np.r_[self._gradient_Q_all, gradient_Q.reshape(1, len(gradient_Q))]

            if self.debug and False:
                print("Gradients:")
                print(self._gradient_V_all)

            # Compute the exponents of gamma*lambda, which go from n_trace_length-1 down to 0
            # starting with the oldest gradient.
            # Note that the trace length is the same for both V and Q,
            # as it is simply the number of rows in _gradient_V_all, which coincides with the number of rows in _gradient_Q_all.
            if gradient_Q is not None:
                assert self._gradient_V_all.shape[0] == self._gradient_Q_all.shape[0]
            n_trace_length = self._gradient_V_all.shape[0]
            exponents = np.array( range(n_trace_length-1, -1, -1) ).reshape(n_trace_length, 1)
                ## The exponents are e.g. (3, 2, 1, 0) when n_trace_length = 4
                ## Note that we reshape the exponents as a column vector because we need to use it
                ## as the exponents of gamma*lambda which will multiply the respective row of the historical gradients
                ## i.e.:
                ## (gamma*lambda)**exponents[0] multiplies row 0 of the historical gradients
                ## (gamma*lambda)**exponents[1] multiplies row 1 of the historical gradients
                ## and so forth...

            # New eligibility trace using the latest computed lambda as weight for ALL past time steps
            self._z_V = np.sum( (self.gamma * lmbda)**exponents * self._gradient_V_all, axis=0 )
            self._z_V_all = np.r_[self._z_V_all, self._z_V.reshape(1, len(self._z_V))]
            if gradient_Q is not None:
                self._z_Q = np.sum( (self.gamma * lmbda)**exponents * self._gradient_Q_all, axis=0 )
                self._z_Q_all = np.r_[self._z_Q_all, self._z_Q.reshape(1, len(self._z_Q))]

            if self.debug and False:
                print("Exponents: {}".format((self.gamma * lmbda)**exponents))

        if self.debug and False:
            print("Z's & lambda's by state:")
            print(self._z_V_all)
            print(self._lambdas_in_episode)

    def _store_lambdas_in_episode(self):
        if self.debug:
            print("lambdas in episode {}:".format(self.episode))
            print(self._lambdas_in_episode)
        self._all_lambdas_by_episode += [[self._lambdas_in_episode[s] for s in self.env.getAllStates()]]

        # Store the (average) _lambdas by episode
        if self.debug:
            print("lambdas in episode {}".format(self.episode))
            _dict_numpy_options = set_numpy_options()
            print(np.c_[self.states[:-1], self._lambdas])
            reset_numpy_options(_dict_numpy_options)
        self.lambda_mean_by_episode += [np.mean(self._lambdas)]

    def compute_lambda_statistics_by_state(self):
        "Computes the number, mean, and standard deviation of historical values of lambda by state"
        lambdas_n = self._all_lambdas_n
        lambdas_mean = [S/n for n, S in zip(self._all_lambdas_n, self._all_lambdas_sum)]
        lambdas_std = [np.sqrt( (S2 - S**2/n) / (n - 1) )
                                for n, S, S2 in zip(self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2)]
        return lambdas_n, lambdas_mean, lambdas_std

    # GETTERS
    def getLambdasByEpisodeAndState(self):
        return self._all_lambdas_by_episode

    # PLOTTING FUNCTIONS
    def plot_info(self, episode, nepisodes):
        """
        Plots selected information about the experiment run.

        Arguments:
        episode: int
            Number of episode of interest for the plot.

        nepisodes: int
            Number of total episodes to run or already run

        what: str
            What to plot. Possible values are:
            - "boxplots": for a boxplot of lambdas by state for different ranges of episode number
                (e.g. early learning episodes vs. late learning episodes) divided into 4 groups).
            - "average": for a plot of average lambda by state over all the episodes run.
        """

        # Finalize setup of all possible graphs created here
        def finalize_plot_1D(ax, state_counts, title="Lambda by state for selected episodes / State count distribution"):
            ax.set_title(title)

            ax.set_xticks(self.env.getAllStates())
            ax.set_xlim((0, self.env.getNumStates()-1))
            ax.set_ylim((0, 1.02))
            ax.set_ylabel("Lambda")
            ax.tick_params(axis='y', colors="orange")
            ax.yaxis.label.set_color("orange")

            # State count distribution
            ax1sec = ax.twinx()
            ax1sec.bar(self.env.getAllStates(), state_counts, color="blue", alpha=0.2)
            ax1sec.tick_params(axis='y', colors="blue")
            ax1sec.yaxis.label.set_color("blue")
            ax1sec.set_ylabel("State count")
            plt.sca(ax) # Go back to the primary axis

        def finalize_plot_2D(ax, state_lambdas_2D, state_counts_2D, fontsize=12, title="Lambda by state for selected episodes / State count distribution"):
            ax.set_title(title)

            (nx, ny) = state_counts_2D.shape
            # Adjust the font size according to the image shape
            # It is assumed that the unmodified fontsize works fine for a 5x5 grid
            fontsize = int( np.min((5/nx, 5/ny)) * fontsize )
            for y in range(ny):
                for x in range(nx):
                    if state_lambdas_2D is None:
                        ax.text(y, x, "N={}".format(int(state_counts_2D[x,y])),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
                    elif state_counts_2D is None:
                        ax.text(y, x, "{:.3f}".format(state_lambdas_2D[x,y]),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
                    else:
                        ax.text(y, x, "{:.3f}\n (N={})".format(state_lambdas_2D[x,y], int(state_counts_2D[x,y])),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

        #---- Plot of state-dependent lambdas
        # Option 1: Compute the mean and standard deviation of lambdas by state for last episode 
        #nstates = self.env.getNumStates()
        #states = np.arange(nstates)
        #lambda_sum = np.zeros_like(states, dtype=float)
        #lambda2_sum = np.zeros_like(states, dtype=float)
        #n = np.zeros_like(states)
        #for i, lmbda in enumerate(self._lambdas):
        #    s = self.states[i]
        #    lambda_sum[s] += lmbda
        #    lambda2_sum[s] += lmbda**2
        #    n[s] += 1
        #    print("i: {}, state: {}, lambda: {}, lambda_sum: {}, n: {}".format(i, s, lmbda, lambda_sum[s], n[s]))
        #lambda_mean = np.nan * np.ones_like(states, dtype=float)
        #lambda_std = np.nan * np.ones_like(states, dtype=float)
        #for s in range(self.env.getNumStates()):
        #    if n[s] > 0:
        #        lambda_mean[s] = lambda_sum[s] / n[s]
        #        if n[s] > 1:
        #            lambda_std[s] = np.sqrt( ( lambda2_sum[s] - n[s] * lambda_mean[s]**2 ) / (n[s] - 1) )
        #ax1.plot(states, lambda_mean, '.-', color="orange")
        #ax1.errorbar(states, lambda_mean, yerr=lambda_std, capsize=4, color="orange")
        #ax1.set_title("Mean and StdDev of state-dependent lambdas at the last episode")

        # Option 2: Use the statistics tracked during experiment to plot lambda statistics by state 
        #(lambdas_n, lambdas_mean, lambdas_std) = self.compute_lambda_statistics_by_state()
        #print("lambda statistics by state:")
        #with np.printoptions(precision=3, suppress=True):
        #    print(np.c_[self.env.getAllStates(), lambdas_n, lambdas_mean, lambdas_std])
        #ax1.plot(self.env.getAllStates(), lambdas_mean, '.-', color="orange")
        #ax1.errorbar(self.env.getAllStates(), lambdas_mean, yerr=lambdas_std, capsize=4, color="orange")

        if nepisodes < 4:
            print("NOTE: No plots by episode number group are generated: at least 4 episodes are needed.")
            return

        # Option 3: Plot distribution of lambdas by range of episode numbers
        # (to see how lambdas by state evolve with time)
        # Define 4 groups of episode numbers range 
        episodes_to_show = lambda nepisodes, nplots: set( list( range(0, nepisodes, max(1, int( (nepisodes-1)/(nplots-1) ))) ) + [nepisodes-1] )
        episodes_to_consider = lambda start, stop: range(start, stop+1)

        #for e in episodes_to_show(nepisodes, nplots):
        #    print("Plotting lambdas for episode {}:".format(e))
        #    print(self._all_lambdas_by_episode[e])
        #    ax1.violinplot(self._all_lambdas_by_episode[e], showmeans=True)
        lambdas_by_state = [[] for _ in self.env.getAllStates()]
        for e in episodes_to_consider(0, nepisodes-1):
            for s in self.env.getAllStates():
                lambdas_by_state[s] += self._all_lambdas_by_episode[e][s]
        states2plot = [s for s in self.env.getNonTerminalStates() if self._state_counts_over_all_episodes[s] > 0] 
        #print("lambdas_by_state for plotting:")
        #print([lambdas_by_state[s] for s in states2plot])

        #plt.figure()
        #ax1 = plt.gca()
        # Note: the violinplot() function does NOT accept an empty list for plotting nor NaN values
        # (when at least a NaN value is present, nothing is shown for the corresponding group!)   
        #plotting.violinplot(ax1, [lambdas_by_state[s] for s in states2plot], positions=states2plot,
                            #color_body="orange", color_lines="orange", color_means="red")
        #finalize_plot(ax1, self.env.getStateCounts())
 
        fig = plt.figure()
        nplots = 4
        axes = fig.subplots(2, int(nplots/2))
        if self.plotwhat == "average":
            # Create a new figure to show the distribution of state counts
            # as we cannot plot them in the same figure as the lambda values
            fig_counts = plt.figure()
            axes_counts = fig_counts.subplots(2, int(nplots/2))

        lambda_mean = np.sum( [S for S in self._all_lambdas_sum] ) / np.sum( [n for n in self._all_lambdas_n] )
        nvisits_min = np.min( [n for n in self._all_lambdas_n] )
        nvisits_mean = np.mean( [n for n in self._all_lambdas_n] )
        nvisits_max = np.max( [n for n in self._all_lambdas_n] )
        print("Average lambda over all steps and episodes: {:.2f}".format(lambda_mean))
        print("# visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})".format(nvisits_min, nvisits_mean, nvisits_max))

        episode_step = max(1, int(nepisodes / nplots))      # Ex: int(50/4) = 48/4 = 12 
        for idx_ax, ax in enumerate(axes.reshape(nplots)):
            episode_begin = idx_ax*episode_step             # Ex: 0, 12, 24, 36
            episode_end = episode_begin + episode_step - 1 if idx_ax < nplots - 1 else nepisodes - 1
                ## Ex: 11, 23, 35, 49 (for the last block of episodes, include ALL remaining episodes)
            #print("Plotting lambdas for episodes from {} to {}...".format(episode_begin+1, episode_end+1))
            lambdas_by_state = [[] for _ in self.env.getAllStates()]
            nvisits_by_state = np.zeros(self.env.getNumStates())
            for e in episodes_to_consider(episode_begin, episode_end):
                for s in self.env.getAllStates():
                    lambdas_by_state[s] += self._all_lambdas_by_episode[e][s]
                    nvisits_by_state[s] += len(self._all_lambdas_by_episode[e][s])
            states2plot = [s for s in self.env.getNonTerminalStates() if nvisits_by_state[s] > 0]
            #print("lambdas_by_state for plotting:")
            #print([lambdas_by_state[s] for s in states2plot])
            lambda_min_episodes = np.min( [np.min(lambdas_by_state[s]) for s in states2plot] )
            lambda_mean_episodes = np.sum( [np.sum(lambdas) for lambdas in lambdas_by_state] ) / np.sum( [n for n in nvisits_by_state] )
            lambda_max_episodes = np.max( [np.max(lambdas_by_state[s]) for s in states2plot] )
            nvisits_min_episodes = np.min( [n for n in nvisits_by_state] )
            nvisits_mean_episodes = np.mean( [n for n in nvisits_by_state] )
            nvisits_max_episodes = np.max( [n for n in nvisits_by_state] )
            print("\nLambdas over episodes {} thru {}: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                  .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))
            print("# visits over episodes {} thru {}: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                   .format(episode_begin+1, episode_end+1, nvisits_min_episodes, nvisits_mean_episodes, nvisits_max_episodes))

            if self.plotwhat == "boxplots":
                # Note: the violinpot() function does NOT accept an empty list for plotting nor NaN values
                # (when at least a NaN value is present, nothing is shown for the corresponding group!)
                plotting.violinplot(ax, [lambdas_by_state[s] for s in states2plot], positions=states2plot,
                                    color_body="orange", color_lines="orange", color_means="red")
                finalize_plot_1D(ax, nvisits_by_state, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                 .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))
                fig.suptitle("Lambdas distribution by state for different time periods of the experiment")
            elif self.plotwhat == "average":
                # Average lambda values by state
                lambda_mean_by_state = [np.mean(lambdas_by_state[s]) if s in states2plot
                                                                     else np.nan
                                                                     for s in self.env.getAllStates()]
                lambda_se_by_state = [np.std(lambdas_by_state[s]) / np.sqrt(nvisits_by_state[s]) if s in states2plot
                                                                                                 else np.nan
                                                                                                 for s in self.env.getAllStates()]
                if self.env.getDimension() == 2:
                    # Prepare data to plot 
                    shape = self.env.getShape()
                    lambda_mean_by_state_2D = np.asarray(lambda_mean_by_state).reshape(shape)
                    nvisits_by_state_2D = np.asarray(nvisits_by_state).reshape(shape)

                    # Display the 2D images
                    # --lambdas distribution
                    colormap = cm.get_cmap("Oranges")
                    colornorm = plt.Normalize(vmin=0.0, vmax=1.0)
                    ax.imshow(lambda_mean_by_state_2D, cmap=colormap, norm=colornorm)
                    finalize_plot_2D(ax, lambda_mean_by_state_2D, nvisits_by_state_2D, fontsize=self.fontsize, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                 .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))

                    # --state count distribution
                    colormap = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=np.min(nvisits_by_state), vmax=np.max(nvisits_by_state))
                    axc = axes_counts[int(idx_ax/2), idx_ax%2]
                    axc.imshow(nvisits_by_state_2D, cmap=colormap, norm=colornorm)
                    finalize_plot_2D(axc, None, nvisits_by_state_2D, fontsize=int(self.fontsize*1.5), title="Episodes {} thru {}\nstate count: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                                 .format(episode_begin+1, episode_end+1, nvisits_min_episodes, nvisits_mean_episodes, nvisits_max_episodes))
                else:
                    # For 1D or dimensions higher than 2, just plot lambdas in terms of the 1D state numbers
                    ax.errorbar(states2plot, lambda_mean_by_state, yerr=lambda_se_by_state, capsize=4, color="orange")
                    finalize_plot_1D(ax, nvisits_by_state, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                     .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))

                fig.suptitle("Lambda by state averaged over episodes in different time periods of the experiment" \
                             "\n(overall lambda average = {:.2f})".format(lambda_mean))
                fig_counts.suptitle("State count distribution over episodes in different time periods of the experiment")
