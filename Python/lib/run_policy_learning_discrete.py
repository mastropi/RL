# 24-Aug-2025
# SCRIPT EXPECTED TO BE INCLUDED FROM ANOTHER SCRIPT THAT DEFINES ALL VARIABLES THAT HERE APPEAR AS UNDEFINED! (typically, the caller script is test/tests.py)
# This is a QUICK WORKAROUND for not duplicating code when setting up the FVAC vs. TDAC experiments on labyrinths with increasing size

print()
print(f"********************************************************")
print(f"* PROCESSING LEARNING METHOD {learning_method.upper()}")
print(f"********************************************************")

simulator_value_functions = simulators[learning_method]
learning_method_type = learning_method[:9]  # This makes e.g. "values_fvos" become "values_fv"

#--- Execution parameters that depend on the learning method
# (2025/06/22) IMPORTANT NOTE ABOUT USING THE ADVANTAGE FUNCTION TO UPDATE THE POLICY:
# It was observed in the labyrinth problem that the learning curves (average reward) are more UNSTABLE when learning using the advantage A(s,a) than the action function Q(s,a)
# The instability in principle can be reduced by reducing the learning rate for the actor (policy_learning_rate) from e.g. 10.0 to 1.0, but not really sure about its effect.
use_advantage = not (learning_method == "values_fvos")  # Set this to True if we want to use the advantage function learned as the TD error, instead of using the advantage function as the difference between the estimated Q(s,a) and the estimated V(s) (where the average reward cancels out)
policy_learning_rate = 1.0 if is_NPG and use_advantage else 10.0 if is_NPG and not use_advantage else 0.05  # if policy_learning_mode == "online" else 0.05 #0.01 #0.1
if use_function_approximation:
    # When using function approximations for V(s) and Q(s,a), we should use a not too large policy learning rate in order to avoid too large changes in the policy
    # that tend to generate the same policy for all states at once... without nuance on the different actions to take at each state.
    # Take the example of a gridworld, if the policy learning rate is small enough, after initial policy learning steps where action probabilities tend to be very similar across cells,
    # the policy then starts drifting to a nuanced policy that is different for different cells (e.g. from t_learn > 6 when policy_learning_rate = 0.1 in 6x8 labyrinth without wind).
    policy_learning_rate /= 10
#--- Execution parameters that depend on the learning method


#-- Log file
prefix = f"ActorCritic_{env_type.name.lower()}_{shape_str}_"
suffix = f"_{learning_method}"
# Open log file if one requested and show method being run
if log:
    dt_start, stdout_sys, stderr_sys, fh_log, _logfile_not_used = log_file_open(logsdir, subdir="", prefix=prefix, suffix=suffix, use_datetime=True)
#-- Log file


print("******")
print(f"Running {learning_method.upper()} method for value functions estimation.")
print(f"A NOMINAL MAXIMUM of {max_time_steps_benchmark} steps will be allowed during the simulation.")
print("******")

# Store the execution parameters in a dictionary
params_exec = dict([(k, eval(k)) for k in [  # --- Environment
                                            'env_type',
                                            'env_type_name',  # We also store the env_type_name because of errors generated when saving to pickle (can't pickle Environment enum)
                                            'env_shape',
                                            'entry_state',
                                            'exit_state',
                                            'wind_dict',
                                            'seed_obstacles',
                                            # --- Learning
                                            'learning_method',
                                            'learning_task',
                                            'learning_criterion',
                                            # --- Absorption set
                                            'estimate_absorption_set_at_every_step',
                                            'update_absorption_set_with_fv_visits',
                                            'soft_killing',
                                            # --- Actor
                                            'policy_learning_mode',
                                            'is_NPG',
                                            'n_learning_steps',
                                            'n_episodes_per_learning_step',
                                            'max_time_steps_per_policy_learning_episode',
                                            'allow_deterministic_policy',
                                            'use_advantage',
                                            'policy_learning_rate',
                                            'adjust_policy_learning_rate',
                                            'reset_value_functions_at_every_learning_step',
                                            # --- Critic
                                            'use_function_approximation',
                                            'N',
                                            'T',
                                            'M1',
                                            'M2',
                                            'max_time_steps_benchmark',
                                            'alpha_initial',
                                            'adjust_alpha_initial_by_learning_step',
                                            'epsilon_random_action',
                                            'use_average_max_time_steps_in_td_learner',
                                            'use_average_reward_from_previous_step',
                                            'use_fixed_average_reward',
                                            'keep_fv_estimation_of_average_reward_and_stationary_probability_consistent',
                                            'learning_steps_observe',
                                            'verbose_period',
                                            'plot',
                                            'colormap',
                                            # -- Simulation
                                            'seed_base',
                                        ]])

print("\nExecution parameters:")
for param, value in params_exec.items():
    print(f"{param}: {value}")

# Initialize objects that will contain the results by learning step
simulators_all = [None] * nrep  # List with the simulator objects so that we can analyze the learning process at each replication, if needed (e.g. when learning doesn't really work)
state_counts_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()), dtype=int)
if env_type == Environment.Gridworld:
    V_true_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()))
V_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()))
Q_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
A_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
R_all = np.nan * np.ones((nrep, n_learning_steps))  # Average reward (corresponding to EPISODIC learning tasks)
R_long_all = np.nan * np.ones((nrep, n_learning_steps))  # Long-run average reward (corresponding to CONTINUING learning tasks). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
R_long_initial_all = np.zeros((nrep, n_learning_steps))  # Useful for FV only: average reward observed during the initial simulation (useful for ablation study of FV)
R_long_fv_inflated_all = np.nan * np.ones((nrep, n_learning_steps))  # Useful for FV only: compare the average reward inflated by the FV exploration and a sensible estimate of the average reward by FV
R_long_true_all = np.nan * np.ones((nrep, n_learning_steps))  # True Long-run Average reward (CONTINUING learning task) under the policy at the start of each policy learning step
loss_all = np.nan * np.ones((nrep, n_learning_steps))
nsteps_all = np.nan * np.ones((nrep, n_learning_steps), dtype=int)  # Number of value function time steps run per every policy learning step
if state_observe is not None:
    V_state_observe_all = np.empty((nrep, n_learning_steps), dtype=object)  # Store the V(s) values estimated during value function learning at the specified state_observe state in all replications and at all learning steps
KL_all = np.nan * np.ones((nrep, n_learning_steps))  # K-L divergence between two consecutive policies
KL_norm_all = np.nan * np.ones((nrep, n_learning_steps))  # NORMALIZED-by-#known-states K-L divergence between two consecutive policies
alpha_all = alpha_initial * np.ones((nrep, n_learning_steps))  # Initial alpha used at each policy learning step
if use_function_approximation:
    LR_all = simulator_value_functions.getAgent().getLearner().getV().getLearningRate() * np.ones((nrep, n_learning_steps))  # Value function optimizer learning rate at the BEGINNING of each policy learning step
else:
    LR_all = np.nan * np.ones((nrep, n_learning_steps))
LR_policy_all = policy_learning_rate * np.ones((nrep, n_learning_steps))  # Policy optimizer learning rate at each policy learning step
time_elapsed_all = np.nan * np.ones(nrep)  # Execution time for each replication
time_cpu_all = np.nan * np.ones(nrep)  # CPU time for each replication

# Prepare the Actor learner
if learning_method_type == "values_fv":
    # Define the object where we will store the number of steps used by the FV learner of value functions at each policy learning step
    # so that we use the same number for the TD learner of value functions when the TDAC policy learning process is run afterwards.
    # I.e. this assumes that FVAC is run BEFORE TDAC!
    max_time_steps_benchmark_all = np.nan * np.ones((nrep, n_learning_steps))
if learning_method == "all_online":
    # Online Actor-Critic policy learner with TD as value functions learner and value functions learning happens at the same time as policy learning
    learner_ac = LeaActorCriticNN(test_ac.getEnv(), simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy,
                                  optimizer_learning_rate=policy_learning_rate, seed=test_ac.seed, debug=True)
else:
    # Value functions (Critic) are learned separately from the application of the policy and the policy (Actor) may be learned OFFLINE or online
    # IMPORTANT: We pass the policy of the agent stored in the value functions simulator as policy for the Actor-Critic learner so that when the Actor-Critic learner
    # updates the policy, the policy of the agent stored in the value functions simulator is ALSO updated. This is crucial for using the updated policy
    # when learning the value functions at the next policy learning step.
    learner_ac = LeaActorCriticNN(test_ac.getEnv(), simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy,
                                  optimizer_learning_rate=policy_learning_rate, seed=test_ac.seed, debug=True)

time_start = timer()
cpu_start = process_time()
dt_start_filename = get_current_datetime_as_string(format="filename")

for rep in range(nrep):
    #if rep + 1 != 2:
    #    print(f"Replication {rep+1} skipped!")
    #    continue
    seed_rep = seed_base * (rep + 1)
    print(f"\n->>>>>>>>>>>\nRunning replication {rep + 1} of {nrep} (seed={seed_rep})... @{format(get_current_datetime_as_string())}")

    # Reset the policy actor every time a new replication starts
    # Note that this performs a further reset of the policy (besides the one done above by the constructor that creates the learner_ac object),
    # and thus resets the policy to ANOTHER slightly different policy (because of the normally distributed random values around 0 that are set as neural network weights)
    # Note also that the critic learner will be reset by the critic learner just before starting with the simulation
    # (see discrete.Simulator._run_fv() method for FV and discrete.Simulator._run_single_continuing_task() for TD)
    print("Resetting the policy learner and the critic (if any)...")
    learner_ac.reset(reset_value_functions=True, reset_policy=True, initial_policy=initial_policy)
    print(f"Resetting the policy learning rate for the policy to {policy_learning_rate}...")
    learner_ac.setOptimizerLearningRate(policy_learning_rate)
    if not test_ac.getEnv().isStateContinuous():
        print(f"Initial policy (states x actions):\n{learner_ac.getPolicy().get_policy_values()}")

    if simulator_value_functions is not None:
        # Whenever there is a critic, do a couple of resets
        # (recall that no critic is defined for the ALL-online learning method)
        _learner_value_functions_for_critic = simulator_value_functions.getAgent().getLearner()

        # "[OUT]" means "out of the main code", i.e. at the caller of the policy learning process run by simulator_value_functions.run() and learner_ac.learn()
        print(f"[OUT] The average reward stored in learner before starting a new replication and resetting the average reward is: {_learner_value_functions_for_critic.getAverageReward()}")

        # Reset the initial alpha of the TD learner (just in case)
        _learner_value_functions_for_critic.setInitialLearningRate(alpha_initial)

    time_start_rep = timer()
    cpu_start_rep = process_time()
    if learning_method == "all_online":
        for t_learn in range(n_learning_steps):
            seed_learn = seed_rep + t_learn
            print(
                f"\n\n*** Running learning step {t_learn + 1} of {n_learning_steps} (AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn - 1)]}, {1 / R_all[rep, max(0, t_learn - 1)]:.0f} average steps) of "
                f"MAX={max_avg_reward_episodic if policy_learning_mode == 'online' else max_avg_reward_continuing} using {nsteps_all[rep, max(0, t_learn - 1)]} time steps for Critic estimation)... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            print("Learning the VALUE FUNCTIONS and POLICY simultaneously...")
            loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode,
                                                      prob_include_in_train=prob_include_in_train)
            ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`, because the environment may have defined
            ## a different initial state distribution (e.g. a random start in the states outside the absorption set used by the FV learner.

            state_counts_all[rep, t_learn, :] = learner_ac.learner_value_functions.getStateCounts()
            V_all[rep, t_learn, :] = learner_ac.learner_value_functions.getV().getValues()
            Q_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            A_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getA().getValues().reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            R_all[rep, t_learn] = learner_ac.learner_value_functions.getAverageReward()
            # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
            # R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
    else:
        # Keep track of the policy learned so that we can analyze how much it changes after each learning step w.r.t. the previous learning step
        policy_prev = None
        # Keep track of the number of learning steps in which we observe a significant change in the policy so that we can reduce the policy learning rate
        # when NPG is used and thus avoid very large policy updates when some learning has already happened.
        n_learning_steps_with_large_enough_KL = 0
        if plot_policy_update:
            # Initialize the plot of the policy at each policy learning step
            # NOTE: The plot of the absorption set only makes sense for the FIRST REPLICATION because at this point the absorption set has not been re-estimated
            # (the estimation is done at the beginning of the FV process).
            axes_policy = plot_policy(test_ac.getEnv(), learner_ac.getPolicy(), np.nan, state_counts_all[rep, :, :], params_exec, is_problem_2d=problem_2d,
                                      absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                                      t_learn=1, verbose=False)

        start_state_first_episode = None
        t_learn_last_reset_learning_rates = t_learn_min_to_adjust_alpha     # Used when adjust_alpha_initial_by_learning_step = True
        for t_learn in range(n_learning_steps):
            # Set the policy in evaluation mode
            # This is important if we are using dropout layers in the neural network, o.w. the policy output by the model may be incorrect because some connections might be missing
            # when evaluating the policy(a|s) by calling policy.getPolicyForState()!! (this is not the case in evaluation mode because ALL the connections are back during evaluation,
            # even if the connection had been dropped during training).
            learner_ac.getPolicy().getModel().eval()

            if env_type == Environment.Gridworld:
                # Compute or update the true state value function stored in the environment for the current policy
                # (used as reference when plotting the evolution of the estimated state value function V(s) when plot=True)
                # Note that we store the true value function at the beginning of the state value function learning, because the current iteration that is used to learn
                # the state value function will learn it for the current policy (BEFORE its update), therefore we need to compare the learned value function with the true
                # value function also BEFORE the update of the policy.
                V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.getEnv(), learner_ac.getPolicy(), learning_task, learning_criterion)
                # Set NaN to the position of the obstacles, so that it does not affect the average of V(s) across states (which may be used to correct V(s) for comparisons)
                for s in test_ac.getEnv().getObstacleStates():
                    V_true[s] = np.nan
                V_true_all[rep, t_learn, :] = V_true
                R_long_true_all[rep, t_learn] = avg_reward_true

            # Pass a different seed (for the simulator) for each learning step... o.w. we will be using the same seed for them at every learning step!!
            seed_learn = seed_rep + t_learn
            if env_type == Environment.Gridworld:
                print(f"\n\n*** Running learning step {t_learn + 1} of {n_learning_steps} (True average reward under current policy = {avg_reward_true}) "
                      f"(AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn - 1)]} of MAX={max_avg_reward_episodic}, {1 / R_all[rep, max(0, t_learn - 1)]:.0f} average steps) "
                      f"(AVERAGE REWARD STORED In learner = {simulator_value_functions.getAgent().getLearner().getAverageReward()})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            else:
                print(f"\n\n*** Running learning step {t_learn + 1} of {n_learning_steps} "
                      f"(AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn - 1)]} of MAX={max_avg_reward_episodic}, {1 / R_all[rep, max(0, t_learn - 1)]:.0f} average steps) "
                      f"(AVERAGE REWARD STORED In learner = {simulator_value_functions.getAgent().getLearner().getAverageReward()})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            time.sleep(1)  # Wait for a second so that I can easily read the learning step number

            # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (because we don't want to keep history from a earlier learning process on the same learner!)
            reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 0 else True

            # -- Optionally adjust the initial learning rate alpha
            # Compute the K-L divergence of the current policy w.r.t. the previous policy to decide whether to update the initial learning rate or not
            # If the new policy is too different from the previous policy we should NOT update the learning rate because the value functions learned
            # under the previous policy is most likely far away from the true values functions under the new policy.
            # Note: rel_entr(p, q) computes the term p * log(p/q) contributing to the relative entropy or Kullback-Leibler divergence
            # of the "new" distribution p(x) w.r.t. the "old" distribution q(x), which is defined as the expectation (measured over p(x)) of the
            # difference between log(p(x)) and log(q(x), i.e. the p-expected difference in information provided by the new policy p(x) w.r.t. the previous policy q(x)).
            # Hence if p(x) is larger than q(x) the contribution is positive and if it is smaller, the contribution is negative.
            # Therefore a positive K-L divergence typically corresponds to an increase in the probability, from q(x) to p(x), at states x having a larger "new" probability p(x).
            # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
            # Note that we could also use the function scipy.stats.entropy() to compute directly the K-L divergence (i.e. without summing over the rel_entr() values, as done here)
            if not test_ac.getEnv().isStateContinuous():
                policy = learner_ac.getPolicy().get_policy_values()
                KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 0 else 0.0
                KL_distance_norm, _num_states_known_by_agent = compute_normalized_KL_distance(KL_distance, simulator_value_functions.getAgent().getLearner())
                print(
                    f"K-L distance with previous policy: {KL_distance:.4f}, standardized by the # states known by the agent ({_num_states_known_by_agent}): {KL_distance_norm:.6f})")
                KL_all[rep, t_learn] = KL_distance
                KL_norm_all[rep, t_learn] = KL_distance_norm
                if KL_distance_norm > KL_THRESHOLD:
                    n_learning_steps_with_large_enough_KL += 1
                policy_prev = policy.copy()

            if adjust_alpha_initial_by_learning_step:
                # Update the initial learning rate alpha (for VALUE FUNCTIONS, not for the policy) ONLY when:
                # - the learning step is larger than or equal to a minimum.
                # - the policy did NOT change significantly from the previous learning step
                #   (because in that case we can consider the current estimates of the value functions to be associated to the NEW policy as well,
                #    since the policy didn't change much).
                if t_learn + 1 >= t_learn_min_to_adjust_alpha:
                    if KL_distance_norm > KL_THRESHOLD:
                        # Reset the learning rates (alpha for TD and Adam's lr) to the initial ones because the policy changed considerably w.r.t. the previous policy learning step
                        t_learn_last_reset_learning_rates = t_learn
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                        if use_function_approximation:
                            simulator_value_functions.getAgent().getLearner().getV().resetInitialLearningRate()
                    else:
                        # Update the initial learning rats for value functions at this policy learning step to a smaller value than the previous policy learning step
                        # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
                        _t_dump = t_learn - t_learn_last_reset_learning_rates + 1   # +1 so that there is a reduction already from the first learning step after t_learn_last_reset_learning_rates
                        _alpha_initial_at_current_learning_step = alpha_initial / _t_dump   # alpha_initial/10)
                        _lr_decrease_factor = 1 / (1 - 1/_t_dump)  # `1 / (1 - 1/_t_dump)` is how much we need to decrease the CURRENT learning rate by in order to apply a decrease of lr0 / _t_dump where lr0 is the INITIAL learning rate.

                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(_alpha_initial_at_current_learning_step)
                        if use_function_approximation:
                            simulator_value_functions.getAgent().getLearner().getV().decreaseLearningRateBy(_lr_decrease_factor)
                alpha_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
                if use_function_approximation:
                    LR_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getV().getLearningRate()
            print(f"*** INITIAL learning rate for VALUE FUNCTIONS, alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()}" + \
                  (use_function_approximation and f", lr={simulator_value_functions.getAgent().getLearner().getV().getLearningRate()}" or ""), end=" ")
            print(f"(adjustment happens starting at learning step >= {t_learn_min_to_adjust_alpha} AND if the NORMALIZED (by the number of known states) K-L change in policy is AT MOST {KL_THRESHOLD}) ***" if adjust_alpha_initial_by_learning_step else "***")

            if adjust_policy_learning_rate:
                # Update the optimizer learning rate (for the POLICY, applies also when learning the policy via NPG) ONLY when:
                # - the learning step is larger than or equal to a minimum.
                # - the policy CHANGED significantly from the previous learning step
                #   (i.e., if no significant learning happened, it doesn't make too much sense to reduce the policy learning rate, because nothing had been learned so far).
                # The adjustment is done by the number of learning steps where the policy has changed significantly so far, instead of e.g. using t_learn directly,
                # with the objective of avoiding a too agressive reduction of the learning rate which could temper fast enough learning of the policy.
                if t_learn + 1 >= t_learn_min_to_adjust_policy_learning_rate:
                    if not test_ac.getEnv().isStateContinuous():
                        if KL_distance_norm > KL_THRESHOLD:
                            LR_policy_all[rep, t_learn] = policy_learning_rate / n_learning_steps_with_large_enough_KL
                            learner_ac.setOptimizerLearningRate(LR_policy_all[rep, t_learn])
                        else:
                            LR_policy_all[rep, t_learn] = learner_ac.getOptimizerLearningRate()
                    else:
                        # Update the policy learning rate by the number of learning steps used so far
                        LR_policy_all[rep, t_learn] = policy_learning_rate / (t_learn + 1)
                        learner_ac.setOptimizerLearningRate(LR_policy_all[rep, t_learn])
            print(f"*** Optimizer learning rate for the POLICY, lr = {learner_ac.getOptimizerLearningRate()}", end=" ")
            print(f"(adjustment happens starting at learning step >= {t_learn_min_to_adjust_policy_learning_rate} AND if standardized K-L change in policy "
                  f"({KL_distance_norm:.4f}) is LARGER THAN {KL_THRESHOLD}) ***" if adjust_policy_learning_rate else "***")

            # --- 1) CRITIC
            print(f"Learning the CRITIC (at the current policy) using {learning_method.upper()}...")
            # Learn the value functions using the FV simulator
            if learning_method_type == "values_fv":
                if plot and t_learn + 1 in learning_steps_observe and t_learn + 1 != learning_steps_observe[0]:
                    _close_plots = input("Before proceeding, would you like to close all plots generated for all learning steps observed so far (y/n)?")
                    if _close_plots.upper() == "Y":
                        for _fig_number in plt.get_fignums():
                            # Do NOT close the first 2 figures that are created at the beginning of the process showing the absorption set and initial trajectory
                            plt.close(_fig_number) if _fig_number > 2 else None
                if False:
                    # DM-2025/05/29: REMOVED THIS STEP BECAUSE THIS IS NOW FULLY DONE BY THE discrete.Simulator class WHEN (i) calling the learner.reset() method with the reset_auxiliary_info=True argument, and when ALWAYS estimating the absorption set at the first learning step, i.e. when t_learn = 0.
                    # Reset to None the start state distribution for the E(T_A) excursion at the very first learning step so that
                    # we do NOT carry over whatever this distribution was at the end of the previous replication or at the end of the previous execution of the learning process
                    # using this same learner.
                    if t_learn == 0:
                        simulator_value_functions.getAgent().getLearner().setProbasStationaryStartStateET(None)
                        # NEW-2024/10/23: We should estimate the absorption set at the FIRST policy learning step for EVERY replication, as estimating the absorption set is part of the FVAC process!
                        # Otherwise, if estimate_absorption_set_at_every_step = False, FVAC will be stick to use ALWAYS the same absorption set A estimated
                        # by the preparation of the simulation environment in test_optimizers_discretetime.py, and this is not completely fair... In fact, we already observed that
                        # the absorption set A may strongly depend on the seed used (e.g. Labyrinth 6x8 with 50% obstacles and obstacle seed = 4217, with WIND = 0.5, where for the
                        # initial simulation seed of 1317, the absorption set A turns out to be extremely favorable for FVAC... but only for THAT SEED!!)
                        estimate_absorption_set_at_this_step = True
                    else:
                        estimate_absorption_set_at_this_step = estimate_absorption_set_at_every_step
                V, Q, A, state_counts, state_counts_et, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_a, n_events_et, n_events_fv = \
                    simulator_value_functions.run(t_learn=t_learn,
                                                  learning_mode=critic_learning_mode,
                                                  max_time_steps=max_time_steps_fv_overall,
                                                  max_time_steps_for_absorbed_particles_check=max_time_steps_fv_for_all_particles,
                                                  min_prop_absorbed_particles=min_prop_absorbed_particles, stopping_criterion_fv=stopping_criterion_fv,
                                                  min_num_cycles_for_expectations=None,
                                                  # When None, the minimum number of cycles for the estimation of E(T_A) is set by the global variable MIN_NUM_CYCLES_FOR_EXPECTATIONS
                                                  estimate_absorption_set=estimate_absorption_set_at_every_step,
                                                  update_absorption_set_with_fv_visits=update_absorption_set_with_fv_visits, threshold_absorption_set=threshold_absorption_set,
                                                  soft_killing=soft_killing,
                                                  use_average_reward_stored_in_learner=use_average_reward_from_previous_step,
                                                  use_fixed_average_reward=use_fixed_average_reward,
                                                  keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=keep_fv_estimation_of_average_reward_and_stationary_probability_consistent,
                                                  reset_value_functions=reset_value_functions_at_this_step,
                                                  plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
                                                  epsilon_random_action=epsilon_random_action,
                                                  reward_for_exit_states=reward_to_promote_exploration if reward_to_promote_exploration != 0 else None,

                                                  # DM-2025/01: This is used when performing reward shaping, in order to update the policy after the initial exploration has finished (so that reward shaping has actually an effect on the policy, which is the goal of doing reward shaping!)
                                                  learner_policy=learner_ac,

                                                  seed=seed_learn, verbose=False, verbose_period=verbose_period)
                average_reward_initial_exploration = simulator_value_functions.getAgent().getLearner().getAverageRewardInitialExploration()
                average_reward_fv_inflated = simulator_value_functions.getAgent().getLearner().getAverageRewardRaw()
                average_reward_from_critic_estimation = expected_reward  # Note: this is the same information stored in the FV learner, i.e. it would also be returned by calling simulator_value_functions.getAgent().getLearner().getAverageReward()
                nsteps_all[rep, t_learn] = n_events_a + n_events_et + n_events_fv
                max_time_steps_benchmark_all[rep, t_learn] = n_events_a + n_events_et + n_events_fv  # Number of steps to use when running TDAC at the respective learning step
            else:
                # TD learners
                if 'max_time_steps_benchmark_all' in locals() and rep < len(max_time_steps_benchmark_all) and t_learn < len(max_time_steps_benchmark_all[rep, :]) and \
                        max_time_steps_benchmark_all[rep, t_learn] != np.nan:
                    # The FV learner was run before running this TD learner
                    if use_average_max_time_steps_in_td_learner:
                        _max_time_steps = int(np.mean(max_time_steps_benchmark_all[rep, :]))
                    else:
                        # => Use the number of steps used by the FV learner at the current policy learning step (t_learn) as maximum number of steps to allow for the TD learner
                        _max_time_steps = max_time_steps_benchmark_all[rep, t_learn]
                else:
                    # When max_time_steps_benchmark_all is not defined, it means that the number of steps to run the TD learner for is calculated above
                    # and may not be exactly equal to the number of steps the FV learner took at each policy learning step.
                    _max_time_steps = max_time_steps_benchmark
                print(f"*** TD learning will use {_max_time_steps} simulation steps. ***")
                if learning_task == LearningTask.EPISODIC:
                    V, Q, A, state_counts, RMSE, MAPE, learning_info = \
                        simulator_value_functions.run(nepisodes=n_episodes_per_learning_step,
                                                      t_learn=t_learn,
                                                      learning_mode=critic_learning_mode,
                                                      start_state_first_episode=start_state_first_episode,
                                                      max_time_steps=max_time_steps_benchmark,
                                                      max_time_steps_per_episode=max_time_steps_per_policy_learning_episode,
                                                      # max_time_steps_benchmark // n_episodes_per_learning_step,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      epsilon_random_action=epsilon_random_action,
                                                      compute_rmse=(state_observe is not None) or (plot and t_learn + 1 in learning_steps_observe),
                                                      plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                else:
                    V, Q, A, state_counts, RMSE, MAPE, learning_info = \
                        simulator_value_functions.run(t_learn=t_learn,
                                                      start_state_first_episode=start_state_first_episode,
                                                      max_time_steps=_max_time_steps,
                                                      learning_mode=critic_learning_mode,
                                                      estimated_average_reward=simulator_value_functions.getAgent().getLearner().getAverageReward() if use_average_reward_from_previous_step else 0.0,
                                                      use_fixed_average_reward=use_fixed_average_reward,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      epsilon_random_action=epsilon_random_action,
                                                      compute_rmse=(state_observe is not None) or (plot and t_learn + 1 in learning_steps_observe),
                                                      plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                average_reward_from_critic_estimation = simulator_value_functions.getAgent().getLearner().getAverageReward()
                nsteps_all[rep, t_learn] = learning_info['nsteps']
                if state_observe is not None:
                    V_state_observe_all[rep, t_learn] = np.array(learning_info['V_state_observe']) # - np.mean(learning_info['V_state_observe'])

                if start_at_less_frequency_visited_states:
                    # Compute the start state for next learning step as one of the states less visited by the previous exploration, so that we start closer to exploring new states
                    # and are also MORE FAIR in the comparison with FVAC (where the start state is at the boundary of A, which is a set of frequently visited states)
                    # Compute the state distribution and choose the start state for the next learning step uniformly at random from the least visited states
                    _dist_states = pd.Series(simulator_value_functions.getAgent().getLearner().getStates()).value_counts()
                    _less_frequently_visited_states = _dist_states[ _dist_states == min(_dist_states) ].index
                    if len(_less_frequently_visited_states) == 1:
                        start_state_first_episode = _less_frequently_visited_states[0]
                    else:
                        # Choose the start state uniformly at random
                        start_state_first_episode = np.random.choice(_less_frequently_visited_states)

            print(f"Learning step #{t_learn + 1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")
            print(f"Estimated average reward by Critic learning process: {average_reward_from_critic_estimation}")
            state_counts_all[rep, t_learn, :] = state_counts
            V_all[rep, t_learn, :] = V
            if Q is not None:
                # The Q function may not always be estimated (for faster processing, as it is not used in learning an optimal policy --only the advantage is used which depends on V(s))
                Q_all[rep, t_learn, :, :] = Q.reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            A_all[rep, t_learn, :, :] = A.reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())

            # --- 2) ACTOR
            print(f"\nLearning the POLICY {policy_learning_mode.upper()} using estimated {use_advantage and 'ADVANTAGE A(s,a) values' or 'ACTION Q(s,a) values'} ", end=" ")
            # Policy learning
            if is_NPG:
                # Learn using NPG (Natural Policy Gradient)
                if use_advantage:
                    print("\n(Natural Policy Gradient learning using the direct estimation of the advantage function)")
                    learner_ac.learn_natural(A)
                else:
                    if True:
                        # DM-2025/06/21: Learning using the advantage function computed as Q - E(Q over actions) proved catastrophic.
                        # Additionally, using Q(s,a) to learn instead of the advantage function A(s,a) estimated by the learner seems much more stable with smoother learning curves.
                        # Context: Labyrinth 6x8 without wind, after extensive tests and analysis of A(s,a), Q(s,a) and average reward learning curves conducted today.
                        learner_ac.learn_natural(Q)
                    else:
                        # Compute the advantage function as Q - V, where V = policy-weighted-average(Q) over all actions, and then update the policy using that advantage function
                        # Under the AVERAGE reward learning criterion, this makes the average reward (present in both V and Q), cancel out, thus making its estimation useless.
                        # In the FV approach, this is tantamount to leveraging Fleming-Viot JUST as an OVERSAMPLING mechanism, not as an estimator of the long-run expected reward.
                        # However, it has been shown by different sources (see e.g. the 2021 paper by K. Ross and Zhang on the average reward actor-critic learning:
                        # "On-Policy Deep RL for the Average-Reward Criterion" or the lecture by David Silver available in YouTube where he talks about this at one point
                        # saying that using a direct estimate of the Advantage function is better than using Q - V).
                        print("\n(Natural Policy Gradient learning using an INDIRECT estimation of the advantage function as Q(s,a) - policy-weighted-avg(Q)-over-all-actions(s))")
                        _A_as_Q_minus_avgQ = np.zeros_like(A)
                        for s in range(test_ac.getEnv().getNumStates()):
                            _state_value = 0.0
                            for a in range(test_ac.getEnv().getNumActions()):
                                # NOTE: This assumes that the order of the states and actions in the 1D arrays containing Q and A is grouped by state,
                                # i.e. for state 0, the value of all actions are given, then the same for state 1 and so forth.
                                _ind2update = s * test_ac.getEnv().getNumActions() + a
                                _state_value += learner_ac.getPolicy().getPolicyForAction(a, s) * Q[_ind2update]
                                _A_as_Q_minus_avgQ[_ind2update] = Q[_ind2update] - _state_value
                        learner_ac.learn_natural(_A_as_Q_minus_avgQ)
                if not test_ac.getEnv().isStateContinuous():
                    print(f"NEW policy (states x actions):\n{learner_ac.getPolicy().get_policy_values()}")
            else:
                # The policy is modeled with a regular neural network (with hidden layer)
                if policy_learning_mode == "online":
                    # ONLINE with critic provided by the action values learned above
                    print(f"on {n_episodes_per_learning_step} episodes starting at state s={entry_state}, using MAX {max_time_steps_per_policy_learning_episode} steps per episode...")
                    loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state,
                                                              max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=prob_include_in_train,
                                                              # (2024/08/15) This learner is used when the state is continuous (e.g. Mountain Car) and in LeaActorCriticNN.learn() the advantage is computed using
                                                              # the model for V(s), instead of using the tabular values of the advantage function passed here as `advantage_values`.
                                                              # CHECK THE learn() CODE TO SEE IF THE ABOVE IS THE CASE!
                                                              learner_value_functions_critic=simulator_value_functions.getAgent().getLearner(),
                                                              use_advantage=use_advantage,
                                                              advantage_values=A,
                                                              action_values=Q,  # This parameter is not used when use_advantage=False
                                                              expected_reward=average_reward_from_critic_estimation)
                    ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`,
                    ## because the environment may have defined a different initial state distribution, which is used during the learning of the value functions,
                    ## for instance, any randomly selected state outside the absorption set A used by the FV learner.
                    ## Note that the learned value functions are passed as critic to the Actor-Critic policy learner via the `action_values` parameter.
                    R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
                else:
                    # OFFLINE learner where ALL states and actions are swept and the loss computed on all of them using the state distribution as weights
                    print("and estimated probabilities from Critic estimation excursion...")
                    if learning_method_type == "values_fv":
                        prob_states = compute_prob_states(state_counts_et, probas_stationary=probas_stationary)
                    else:
                        prob_states = compute_prob_states(state_counts)
                    loss_all[rep, t_learn] = learner_ac.learn_offline_from_estimated_value_functions(V, A, Q, state_counts, prob_states=prob_states, use_advantage=use_advantage)
                    R_all[rep, t_learn] = average_reward_from_critic_estimation  # ...although here we should store the EPISODIC average reward
                    # (regardless of the learning task type --CONTINUING or EPISODIC), and when the learning task is
                    # CONTINUING, this average_reward_from_critic_estimation is the continuing average reward...
                    # In any case, at this point we don't have easy access to the episodic average reward that we could
                    # use to store here.
                    # NOTE THAT THIS IS NOT THE NPG case: the NPG case is the ABOVE block, before the ELSE that defines this block.
                    _dict_numpy_options = set_numpy_options()
                    print(f"True stationary probabilities:\n{mu.reshape(env_shape)}")
                    print(f"Estimated stationary probabilities:\n{prob_states.reshape(env_shape)}")
                    reset_numpy_options(_dict_numpy_options)
                    if False and (average_reward_from_critic_estimation != 0.0 or t_learn + 1 in learning_steps_observe):
                        def plot_probas(ax, prob_states_2d, fontsize=14, colormap="Blues", color_text="orange"):
                            colors = cm.get_cmap(colormap)
                            ax.imshow(prob_states_2d, cmap=colors)
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                            for x in range(env_shape[0]):
                                for y in range(env_shape[1]):
                                    ax.text(y, x, "{:.3f}".format(prob_states_2d[x, y]), color=color_text, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")

                        ax_true, ax_est, ax_diff = plt.figure().subplots(1, 3)
                        _fontsize = 14
                        _factor_fs = np.min((5 / env_shape[0], 5 / env_shape[1]))
                        plot_probas(ax_true, mu.reshape(env_shape), fontsize=_fontsize * _factor_fs)
                        plot_probas(ax_est, prob_states.reshape(env_shape), fontsize=_fontsize * _factor_fs)
                        plot_probas(ax_diff, (prob_states - mu).reshape(env_shape), fontsize=_fontsize * _factor_fs, colormap="jet")
                        ax_true.set_title("True stationary probability")
                        ax_est.set_title("Estimated stationary probability (based on state counts)")
                        ax_diff.set_title("Difference: estimated - true")
                        plt.suptitle("Comparison between the true and estimated stationary probabilities", fontsize=24)
                        plt.pause(0.1)
                        plt.draw()
                        input("Press ENTER to continue...")

            # Collect a trajectory for plotting purposes (i.e. without learning)
            # Note that this trajectory is also used to compute the episodic average reward (stored in the R_all object) when using NPG or learning the policy offline
            # (in which case no simulation is run under the current policy as is the case in the ONLINE non-NPG learning approach)
            try:
                # 13-Jan-2025: We `try` this deepcopy() because today I got the following NotImplementedError when copying the simulator object when `plot=True`:
                # "File "C:\ProgramData\Anaconda\Anaconda3-v5.1.0-201802\lib\site-packages\matplotlib\transforms.py", line 127,
                # in __copy__ TransformNode instances can not be copied. "
                # "NotImplementedError: TransformNode instances can not be copied. Consider using frozen() instead."
                _simulator = copy.deepcopy(simulator_value_functions)  # We create a copy because we don't want to change the learner object in the simulator eventually used above (e.g. the state counts that are plotted below)
                simulators_all[rep] = _simulator  # Note that, when `_simulator is re-created at the next iteration with the above assignment, the value of `simulators[rep]` stays INTACT (just tested with dummy lists)
            except:
                print("WARNING: The `simulator_value_functions` object could NOT be DEEPCOPied. "
                      "This means that the online exploration after the value functions have been learned will override the trajectory stored in the value function's learner. "
                      "This might affect trajectory plots which might show unexpected results.")
                _simulator = simulator_value_functions
                simulators_all[rep] = _simulator  # In this case the list will contain only the simulator object of the last replication repeated nrep times, unfortunately
            # Generate a trajectory for the current policy, so that we always have a trajectory to plot,
            # regardless of whether we learn the policy via NPG (in which case no trajectory is generated to learn the policy)
            # or whether we learn the policy via regular policy gradient (in which case a trajectory is generated when updating the policy by the learner_ac.learn() call above).
            # Note that we use 1000 time steps, regardless of the value of T above (used for the estimation of E(T_A) in the FVAC learning case). We do so in order to get
            # a reasonable estimation of the average reward, because the value of parameter T may be too small (e.g. T = 100).
            _learner_current_policy, _nsteps, _average_reward = _simulator.run_exploration(t_learn=t_learn, max_time_steps=1000, epsilon_random_action=0.0,
                                                                                           seed=seed_learn + 171317, verbose=False, verbose_period=verbose_period)
            trajectory_under_policy = np.array(_learner_current_policy.getStates())
            if is_NPG or policy_learning_mode != "online":
                # Note that the average reward is NOT estimated by the run_exploration() method called above, therefore we compute it here from the observed rewards
                R_all[rep, t_learn] = np.mean(_average_reward)

            # Store the long-run average reward estimated by the value functions learner used above
            R_long_all[rep, t_learn] = average_reward_from_critic_estimation

            if learning_method_type == "values_fv":
                # Store auxiliary information on the average reward which can help understand the usefulness of the FV simulation (i.e. towards an ablation study)
                R_long_initial_all[rep, t_learn] = average_reward_initial_exploration
                # Store the inflated average reward (inflated by the FV oversampling effect) in order to analyze how sensible is the average reward estimated by FV
                R_long_fv_inflated_all[rep, t_learn] = average_reward_fv_inflated

            # Check if we need to stop learning because the average reward didn't change a bit
            if break_when_no_change and t_learn > 0 and R_all[rep, t_learn] - R_all[rep, t_learn - 1] == 0.0 or \
                    break_when_goal_reached and np.isclose(R_all[rep, t_learn], max_avg_reward_episodic, rtol=0.001):
                print(f"*** Policy learning process stops at learning step t_learn+1={t_learn + 1} because the average reward didn't change a bit from the previous learning step! ***")
                break

            if plot_policy_update:
                # Initialize the plot of the policy at each policy learning step
                plot_policy(test_ac.getEnv(), learner_ac.getPolicy(), R_all[rep, t_learn], state_counts_all[rep, :t_learn + 1, :], params_exec, axes=axes_policy,
                            is_problem_2d=problem_2d,
                            KL_distance=KL_all[rep, t_learn], KL_distance_norm=KL_norm_all[rep, t_learn],
                            absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                            state_value=V_all[rep, t_learn, :],
                            t_learn=t_learn + 1, verbose=False)
    time_elapsed_rep = timer() - time_start_rep
    time_cpu_rep = process_time() - cpu_start_rep
    print(f"<<<<<<<<<< FINISHED replication {rep + 1} of {nrep}... (@{format(get_current_datetime_as_string())}" + ", took {:.1f} min (CPU: {:.1f}))".format(time_elapsed_rep / 60,
                                                                                                                                                             time_cpu_rep / 60))
    time_elapsed_all[rep] = time_elapsed_rep
    time_cpu_all[rep] = time_cpu_rep

    # ------------------ Plots -----------------
    if not plot_policy_update:
        # Show the final policy
        plot_policy(test_ac.getEnv(), learner_ac.getPolicy(), R_all[rep, t_learn], state_counts_all[rep, :t_learn + 1, :], params_exec,
                    is_problem_2d=problem_2d,
                    KL_distance=KL_all[rep, t_learn], KL_distance_norm=KL_norm_all[rep, t_learn],
                    absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                    state_value=V_all[rep, t_learn, :],
                    t_learn=t_learn + 1, verbose=False)

    # Plot loss and average reward for the current replication
    print("\nPlotting...")
    ax_loss = plt.figure(figsize=figsize).subplots(1, 1)
    ax_loss.plot(np.arange(1, n_learning_steps + 1), loss_all[rep, :n_learning_steps], marker='.', color="red")
    # ax_loss.plot(range(1, n_learning_steps+1), alpha_all[rep, :n_learning_steps], '--', color="cyan")
    ax_loss.set_xlabel("Learning step")
    ax_loss.set_ylabel("Loss", color="red")
    # ax_loss.axhline(0, color="red", linewidth=1, linestyle='dashed')
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_loss.legend(["Loss", "alpha0"], loc='upper left')
    ax_R = ax_loss.twinx()
    legend_R = []
    if policy_learning_mode == "online":
        # We learn the policy by doing a final excursion using the current policy and computing the loss
        # => Plot the episodic average reward observed during the ONLINE Actor-Critic excursion
        ax_R.plot(np.arange(1, n_learning_steps + 1), R_all[rep, :n_learning_steps], marker='.', color="green")
        legend_R += ["Average reward (episodic) (AFTER updating policy)"]
        if not np.isnan(max_avg_reward_episodic):
            ax_R.axhline(max_avg_reward_episodic, color="green", linewidth=1)
            legend_R += ["Max. average reward (episodic)"]
    ax_R.plot(np.arange(1, n_learning_steps + 1), R_long_all[rep, :n_learning_steps], marker='.', color="greenyellow")
    legend_R += ["Long-run average reward estimated by value functions learner (BEFORE updating policy)"]
    if not np.isnan(max_avg_reward_continuing):
        ax_R.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
        legend_R += ["Max. average reward (continuing)"]
    ax_R.set_ylabel("Average reward", color="green")
    # For now I have eliminated the plot of the K-L divergence between consecutive learning steps to avoid cluttering
    ax_R.plot(np.arange(1, n_learning_steps + 1), KL_norm_all[rep, :n_learning_steps], color="blue", linewidth=1)
    ax_R.axhline(KL_THRESHOLD, color="blue", linestyle="dashed")
    legend_R += ["Standardized K-L divergence with previous policy", "Standardized K-L threshold for reducing learning rates"]
    ax_R.axhline(0, color="green", linewidth=1, linestyle='dashed')  # color="green" because it refers to the average reward which is plotted in green on the RIGHT axis
    ax_R.legend(legend_R, loc="upper right")
    plt.title(
        f"{learning_method.upper()}" + f"{((' - SOFT' if soft_killing else ' - HARD') + ' killing') if learning_method_type == 'values_fv' else ''} (rep = {rep + 1} of {nrep} replications)" +
        f"\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulators_all[rep].getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
        f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
        f"\nEvolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")
    # ------------------ Plots -----------------

time_end = timer()
cpu_end = process_time()
time_elapsed = time_end - time_start
time_cpu = cpu_end - cpu_start

if log:
    log_file_close(fh_log, stdout_sys, stderr_sys, dt_start)
show_elapsed_time(learning_method, time_elapsed, time_cpu)


############# Store the measures that we would like to compare
dict_test_ac[learning_method] = test_ac
dict_simulator[learning_method] = simulators_all.copy()
dict_loss[learning_method] = loss_all.copy()
dict_R[learning_method] = R_all.copy()
dict_R_long[learning_method] = R_long_all.copy()
dict_R_long_initial[learning_method] = R_long_initial_all.copy()
dict_R_long_fv_inflated[learning_method] = R_long_fv_inflated_all.copy()
dict_R_long_true[learning_method] = R_long_true_all.copy()
dict_V[learning_method] = V_all.copy()
dict_Q[learning_method] = Q_all.copy()
dict_A[learning_method] = A_all.copy()
dict_state_counts[learning_method] = state_counts_all.copy()
dict_nsteps[learning_method] = nsteps_all.copy()
dict_KL[learning_method] = KL_all.copy()
dict_KL_norm[learning_method] = KL_norm_all.copy()
dict_alpha[learning_method] = alpha_all.copy()
dict_LR[learning_method] = LR_all.copy()
dict_LR_policy[learning_method] = LR_policy_all.copy()
dict_time_elapsed[learning_method] = time_elapsed_all.copy()
dict_time_cpu[learning_method] = time_cpu_all.copy()
############# Store the measures that we would like to compare
