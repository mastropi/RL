# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:26:20 2020

@author: Daniel Mastropietro
@description: Reinforcement Learning simulators are defined on a given environment and agent
interacting with the environment.
The environments are assumed to have the following methods defined:
    - reset()
    - step(action) --> the method that takes the given action and updates the state, returning a tuple:
        (next_state, reward, done, info)

The environments of class gym.toy_text.Env satisfy the above conditions.
"""

import runpy
runpy.run_path('../../setup.py')

import copy
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm  # cm is for colormaps (e.g. cm.get_cmap())

from agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
from agents.queues import PolicyTypes
from environments.queues import rewardOnJobRejection_ExponentialCost, Actions, COST_EXP_BUFFER_SIZE_REF

from utils.computing import rmse, compute_job_rates_by_server, generate_min_exponential_time, stationary_distribution_birth_death_process
import utils.plotting as plotting
from queues import QueueMM, Event

# TODO: (2020/05) Rename this class to SimulatorDiscreteEpisodic as it simulates on a discrete (in what sense?) environment running on episodes
class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a discrete environment

    Arguments:
    env: Environment
        Environment that is assumed to have the following getters and setters defined:
        - getNumStates()
        - getNumActions()
        - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
            (responsible for defining the initial state when running the simulation)
        - getState() --> returns the current state of the environment
        - setInitialStateDistribution()

    agent: Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.
    """

    def __init__(self, env, agent, seed=None, debug=False):
#        if not isinstance(env, EnvironmentDiscrete):
#            raise TypeError("The environment must be of type {} from the {} module ({})" \
#                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        # TODO: (2020/04/12) Fix the following check on the type of agent as it does not work...
        # even though the class of agent is Python.lib.agents.GenericAgent
#        if not isinstance(agent, GenericAgent):
#            raise TypeError("The agent must be of type {} from the {} module ({})" \
#                            .format(GenericAgent.__name__, GenericAgent.__module__, agent.__class__))

        self.debug = debug

        self.env = env
        self.agent = agent
        self.seed = seed

        self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the initial state distribution array in case it has been possibly changed
        # (e.g. when we want to define a specific initial state)
        if self._isd_orig is not None:
            self.setInitialStateDistribution(self._isd_orig)

    def reset(self):
        "Resets the simulator"
        # Copy of Initial State Distribution of environment in case we need to change it
        self._isd_orig = None

        # Reset the learner to the first episode state
        self.agent.getLearner().reset(reset_episode=True, reset_value_functions=True)

    def run(self, nepisodes, start=None, seed=None, compute_rmse=False, state_observe=None,
             verbose=False, verbose_period=1,
             plot=False, colormap="seismic", pause=0):
        # TODO: (2020/04/11) Convert the plotting parameters to a dictionary named plot_options or similar.
        # The goal is to group OPTIONAL parameters by their function/concept.
        """Runs an episodic Reinforcement Learning experiment.
        No reset of the learning history is done at the onset of the first episode to run.
        It is assumed that this reset, when needed, is done by the calling function.

        Parameters:
        nepisodes: int
            Length of the experiment: number of episodes to run.

        start: None or int, optional
            Index in the set of states defined by the environment corresponding to the starting state.

        seed: (opt) None or float
            Seed to use for the random number generator for the simulation.
            If None, the seed is NOT set.
            If 0, the seed stored in the object is used.
            This is useful if this method is part of a set of experiments run and we want to keep
            the seed setting that had been done at the offset of the set of experiments. Otherwise,
            if we set the seed again now, the experiments would have always the same outcome.

        compute_rmse: bool, optional
            Whether to compute the RMSE over states (weighted by their number of visits)
            after each episode. Useful to analyze rate of convergence of the estimates.

        state_observe: int, optional
            A state index whose RMSE should be observed as the episode progresses.

        verbose: bool, optional
            Whether to show the experiment that is being run and the episodes for each experiment.

        verbose_period: int, optional
            Every how many episodes per experiment should be displayed.

        plot: bool, optional
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: str, optional
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.

        pause: float, optional
            Number of seconds to wait before updating the plot that shows the evalution
            of the value function estimates.

        Returns: tuple
            Tuple containing the following elements:
                - state value function estimate for each state at the end of the last episode (`nepisodes`).
                - number of visits to each state at the end of the last episode.
                - RMSE (when `compute_rmse` is not None), an array of length `nepisodes` containing the
                Root Mean Square Error after each episode, of the estimated value function averaged
                over all states. Otherwise, None.
                - a dictionary containing additional relevant information, as follows:
                    - 'alphas_at_episode_end': the value of the learning parameter `alpha` for each state
                    at the end of the last episode run.
                    - 'alphas_by_episode': (average) learning parameter `alpha` by episode
                    (averaged over visited states in each episode).
        """
        #--- Parse input parameters
        if plot:
            fig_V = plt.figure()
            colors = cm.get_cmap(colormap, lut=nepisodes)
            if self.env.getDimension() == 1:
                # Plot the true state value function (to have it as a reference already
                plt.plot(self.env.all_states, self.env.getV(), '.-', color="blue")
                if state_observe is not None:
                    fig_RMSE_state = plt.figure()

        # Define initial state
        nS = self.env.getNumStates()
        if start is not None:
            if not (isinstance(start, int) and 0 <= start and start < nS):
                warnings.warn("The `start` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "A start state will be selected based on the initial state distribution of the environment." \
                              .format(start, type(start), nS-1))
            else:
                # Change the initial state distribution of the environment so that
                # the environment resets to start at the given 'start' state.
                self._isd_orig = self.env.getInitialStateDistribution()
                isd = np.zeros(nS)
                isd[start] = 1.0
                self.env.setInitialStateDistribution(isd)

        # Check initial state
        if state_observe is not None:
            if not (isinstance(state_observe, int) and 0 <= state_observe and state_observe < nS):
                warnings.warn("The `state_observe` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "The state whose index falls in the middle of the state space will be observed." \
                              .format(state_observe, type(state_observe), nS-1))
                state_observe = int(nS/2)

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()
        # Reset the simulator (i.e. prepare it for a fresh new simulation with all learning memory erased)
        self.reset()
        if seed:
            if seed != 0:
                self.env.seed(seed)
            else:
                self.env.seed(self.seed)

        RMSE = np.nan*np.zeros(nepisodes) if compute_rmse else None
        if state_observe is not None:
            V = [learner.getV().getValue(state_observe)]
            # Keep track of the number of times the true value function of the state (assumed 0!)
            # is inside its confidence interval. In practice this only works for the mid state
            # of the 1D gridworld with the random walk policy.
            ntimes_inside_ci95 = 0

        if verbose:
            print("Value function at start of experiment: {}".format(learner.getV().getValues()))
        for episode in range(nepisodes):
            self.env.reset()
            done = False
            if verbose and np.mod(episode, verbose_period) == 0:
                print("Episode {} of {} running...".format(episode+1, nepisodes), end=" ")
                print("(agent starts at state: {}".format(self.env.getState()), end=" ")
            if self.debug:
                print("\n[DEBUG] Starts at state {}".format(self.env.getState()))
                print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

            # Time step in the episode (the first time step is t = 0
            t = -1
            while not done:
                t += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                action = policy.choose_action()
                next_state, reward, done, info = self.env.step(action)
                #if self.debug:
                #    print("| t: {} ({}) -> {}".format(t, action, next_state), end=" ")

                if self.debug and done:
                    print("--> [DEBUG] Done [{} iterations] at state {} with reward {}".
                          format(t+1, self.env.getState(), reward))
                    print("\t[DEBUG] Updating the value function at the end of the episode...")

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                learner.learn_pred_V(t, state, action, next_state, reward, done, info)
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V += [learner.getV().getValue(state_observe)]

                if self.debug:
                    print("-> [DEBUG] {}".format(self.env.getState()), end=" ")

            if verbose and np.mod(episode, verbose_period) == 0:
                print(", agent ENDS at state: {})".format(self.env.getState()))

            if compute_rmse:
                if self.env.getV() is not None:
                    RMSE[episode] = rmse(self.env.getV(), learner.getV().getValues())#, weights=learner.getStateCounts())

            if plot and np.mod(episode, verbose_period) == 0:
                # Plot the estimated value function at the end of the episode
                if self.env.getDimension() == 2:
                    plt.figure(fig_V.number)
                    (ax_V, ax_C) = fig_V.subplots(1, 2)
                    shape = self.env.getShape()
                    terminal_rewards = [r for (_, r) in self.env.getTerminalStatesAndRewards()]

                    state_values = np.asarray(learner.getV().getValues()).reshape(shape)
                    colornorm = plt.Normalize(vmin=np.min(terminal_rewards), vmax=np.max(terminal_rewards))
                    ax_V.imshow(state_values, cmap=colors, norm=colornorm)

                    state_counts = np.asarray(learner.getStateCounts()).reshape(shape)
                    colors_count = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=0, vmax=np.max(state_counts))
                    ax_C.imshow(state_counts, cmap=colors_count, norm=colornorm)

                    fig_V.suptitle("Episode {} of {}".format(episode, nepisodes))
                    if pause > 0:
                        plt.pause(pause)
                    plt.draw()
                else:
                    #print("episode: {} (T={}), color: {}".format(episode, t, colors(episode/nepisodes)))
                    plt.figure(fig_V.number)
                    plt.plot(self.env.all_states, learner.getV().getValues(), linewidth=0.5, color=colors(episode/nepisodes))
                    plt.title("Episode {} of {}".format(episode+1, nepisodes))
                    if pause > 0:
                        plt.pause(pause)
                    plt.draw()
                    #fig.canvas.draw()

                    if state_observe is not None:
                        plt.figure(fig_RMSE_state.number)

                        # Compute quantities to plot
                        RMSE_state_observe = rmse(np.array(self.env.getV()[state_observe]), np.array(learner.getV().getValue(state_observe)))
                        se95 = 2*np.sqrt( 0.5*(1-0.5) / (episode+1))
                        # Only count falling inside the CI starting at episode 100 so that
                        # the normal approximation of the SE is more correct.
                        if episode + 1 >= 100:
                            ntimes_inside_ci95 += (RMSE_state_observe <= se95)

                        # Plot of the estimated value of the state
                        plt.plot(episode+1, learner.getV().getValue(state_observe), 'r*-', markersize=3)
                        #plt.plot(episode, np.mean(np.array(V)), 'r.-')
                        # Plot of the estimation error and the decay of the "confidence bands" for the true value function
                        plt.plot(episode+1, RMSE_state_observe, 'k.-', markersize=3)
                        plt.plot(episode+1,  se95, color="gray", marker=".", markersize=2)
                        plt.plot(episode+1, -se95, color="gray", marker=".", markersize=2)
                        # Plot of learning rate
                        #plt.plot(episode+1, learner._alphas[state_observe], 'g.-')

                        # Finalize plot
                        ax = plt.gca()
                        ax.set_ylim((-1, 1))
                        yticks = np.arange(-10,10)/10
                        ax.set_yticks(yticks)
                        ax.axhline(y=0, color="gray")
                        plt.title("Value and |error| for state {} - episode {} of {}".format(state_observe, episode+1, nepisodes))
                        plt.legend(['value', '|error|', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)'])
                        if episode + 1 == nepisodes:
                            # Show the true value function coverage as x-axis label
                            ax.set_xlabel("% Episodes error is inside 95% Confidence Interval (+/- 2*SE) (for episode>=100): {:.1f}%" \
                                          .format(ntimes_inside_ci95/(episode+1 - 100 + 1)*100))
                        plt.legend(['value', '|error|', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)'])
                        plt.draw()

            if isinstance(learner, LeaTDLambdaAdaptive) and episode == nepisodes - 1:
                learner.plot_info(episode, nepisodes)

            # Reset the learner for the next iteration
            # (WITHOUT resetting the value functions nor the episode counter)
            learner.reset(reset_episode=False, reset_value_functions=False)

        # Comment this out to NOT show the plot right away
        # in case the calling function adds a new plot to the graph generated here
        if plot:
            if self.env.getDimension() == 2:
                # Plot the state counts in a separate image
                fig_C = plt.figure()
                plt.figure(fig_C.number)

                state_counts_min = np.min( learner.getStateCounts() )
                state_counts_mean = np.mean( learner.getStateCounts() )
                state_counts_max = np.max( learner.getStateCounts() )

                shape = self.env.getShape()
                state_counts = np.asarray(learner.getStateCounts()).reshape(shape)
                colors_count = cm.get_cmap("Blues")
                colornorm = plt.Normalize(vmin=0, vmax=np.max(state_counts))
                plt.imshow(state_counts, cmap=colors_count, norm=colornorm)
                # Font size factor
                fontsize = 14
                factor_fs = np.min((5/shape[0], 5/shape[1]))
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        # Recall the x axis corresponds to the columns of the matrix shown in the image
                        # and the y axis corresponds to the rows
                        plt.text(x, y, "{:.0f}".format(state_counts[y,x]),
                                 fontsize=fontsize*factor_fs, horizontalalignment='center', verticalalignment='center')

                plt.title("State counts by state\n# visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                          .format(state_counts_min, state_counts_mean, state_counts_max))
            else:
                # Add the state counts to the plot of the state value function
                #plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
                plt.figure(fig_V.number)
                ax = plt.gca()
                ax2 = ax.twinx()    # Create a secondary axis sharing the same x axis
                ax2.bar(self.env.all_states, learner.getStateCounts(), color="blue", alpha=0.3)
                plt.sca(ax) # Go back to the primary axis
                #plt.figure(fig_V.number)

        self.finalize_run()

        return  learner.getV().getValues(), learner.getStateCounts(), RMSE, \
                {# Value of alpha for each state at the end of the LAST episode run
                 'alphas_at_episode_end': learner._alphas,
                 # (Average) alpha by episode (averaged over visited states in the episode)
                 'alphas_by_episode': learner.alpha_mean_by_episode
                 }

    def finalize_run(self):
        # Restore the initial state distribution of the environment (for the next simulation)
        # Note that the restore step done in the __exit__() method may NOT be enough, because the Simulator object
        # may still exist once the simulation is over.
        if self._isd_orig is not None:
            self.env.setInitialStateDistribution(self._isd_orig)
            self._isd_orig = None

        # Set the value of terminal states to their reward, both the True values and estimated values
        # (just to make plots of the state value function more understandable, specially in environments > 1D)
        for s, r in self.env.getTerminalStatesAndRewards():
            self.agent.getLearner().getV().setWeight(s, r)
            if self.env.getV() is not None:
                self.env.getV()[s] = r

    def simulate(self, nexperiments, nepisodes, start=None, verbose=False, verbose_period=1):
        """Simulates the agent interacting with the environment for a number of experiments and number
        of episodes per experiment.

        Parameters:
        nexperiments: int
            Number of experiments to run.

        nepisodes: int
            Number of episodes to run per experiment.

        start: int, optional
            Index of the state each experiment should start at.
            When None, the starting state is picked randomly following the initial state distribution
            of the environment.

        verbose: bool, optional
            Whether to show the experiment that is being run and the episodes for each experiment.

        verbose_period: int, optional
            Every how many episodes per experiment should be displayed.

        Returns: tuple
            Tuple containing the following elements:
                - Avg(N): The average number of visits to each state over all experiments
                - Avg(RMSE): Root Mean Square Error averaged over all experiments
                - SE(RMSE): Standard Error of the average RMSE
                - Episodic RMSE: array containing the Root Mean Square Error by episode averaged
                over all experiments.
                - a dictionary containing additional relevant information, as follows:
                    - 'alphas_at_episode_end': the value of the learning parameter `alpha` for each state
                    at the end of the last episode in the last experiment.
                    - 'alphas_by_episode': (average) learning parameter `alpha` by episode
                    (averaged over visited states in each episode) in the last experiment.
        """

        if not (isinstance(nexperiments, int) and nexperiments > 0):
            raise ValueError("The number of experiments must be a positive integer number ({})".format(nexperiments))
        if not (isinstance(nepisodes, int) and nepisodes > 0):
            raise ValueError("The number of episodes must be a positive integer number ({})".format(nepisodes))

        N = 0.      # Average number of visits to each state over all experiments
        RMSE = 0.   # RMSE at the end of the experiment averaged over all experiments
        RMSE2 = 0.  # Used to compute the standard error of the RMSE averaged over all experiments
        RMSE_by_episodes = np.zeros(nepisodes)  # RMSE at each episode averaged over all experiments
        RMSE_by_episodes2 = np.zeros(nepisodes) # Used to compute the standard error of the RMSE by episode

        # IMPORTANT: We should use the seed of the environment and NOT another seed setting mechanism
        # (such as np.random.seed()) because the EnvironmentDiscrete class used to simulate the process
        # (defined in the gym package) sets its seed to None when constructed.
        # This seed is then used for the environment evolution realized with the reset() and step() methods.
        [seed] = self.env.seed(self.seed)
        for exp in np.arange(nexperiments):
            if verbose:
                print("Running experiment {} of {} (#episodes = {})..." \
                      .format(exp+1, nexperiments, nepisodes), end=" ")
            V, N_i, RMSE_by_episodes_i, learning_info = self.run(nepisodes=nepisodes, start=start, seed=None,
                                                                 compute_rmse=True, plot=False,
                                                                 verbose=verbose, verbose_period=verbose_period)
            N += N_i
            # RMSE at the end of the last episode
            RMSE_i = RMSE_by_episodes_i[-1]
            RMSE += RMSE_i
            RMSE2 += RMSE_i**2
            RMSE_by_episodes += RMSE_by_episodes_i
            RMSE_by_episodes2 += RMSE_by_episodes_i**2
            if verbose:
                print("\tRMSE(at end of experiment) = {:.3g}".format(RMSE_i))

        N_mean = N / nexperiments
        RMSE_mean = RMSE / nexperiments
        RMSE_se = np.sqrt( ( RMSE2 - nexperiments*RMSE_mean**2 ) / (nexperiments - 1) ) \
                / np.sqrt(nexperiments)
            ## The above, before taking sqrt is the estimator of the Var(RMSE) = (SS - n*mean^2) / (n-1)))
        RMSE_by_episodes_mean = RMSE_by_episodes / nexperiments
        RMSE_by_episodes_se = np.sqrt( ( RMSE_by_episodes2 - nexperiments*RMSE_by_episodes_mean**2 ) / (nexperiments - 1) ) \
                            / np.sqrt(nexperiments)

        return N_mean, RMSE_mean, RMSE_se, RMSE_by_episodes_mean, RMSE_by_episodes_se, learning_info


class SimulatorQueue(Simulator):
    """
    Simulator class that runs a Reinforcement Learning simulation on a given Queue environment `env` and an `agent`
    for as many queue transition iterations and learning iterations as specified in the `dict_nsteps` dictionary.

    Arguments:
    env: Environment
        Queue environment is assumed to have the following methods defined:
        - reset()
        - getNumServers()
        - getJobClassRates()
        - getServiceRates()
        - getJobClass()
        - getQueueState()
        - getState()
        - setJobClass()

    agent: Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.

    dict_nsteps: dict
        Dictionary containing the maximum number of queue steps and learning steps.
        It should contain the following keys:
        - 'queue': number of iterations that the queue should undergo
        - 'learn': number of learning steps (of the value functions and policies)
    """

    def __init__(self, env, agent, dict_nsteps, seed=None, debug=False):
        super().__init__(env, agent, seed, debug)
        self.dict_nsteps = dict_nsteps

        # Job-rates by server (assuming jobs are pre-assigned to servers)
        self.job_rates_by_server = compute_job_rates_by_server(self.env.getJobClassRates(),
                                                               self.env.getNumServers(),
                                                               self.agent.getAssignmentPolicy().getProbabilisticMap())

        # Learners for the value function and the policy
        self.learnerV = self.agent.getLearnerV()
        self.learnerP = self.agent.getLearnerP()

        # Storage of learning history
        self.thetas = []            # Theta before the update by the policy learning step
        self.thetas_updated = []    # Theta AFTER the update by the policy learning step.
        self.V = []                 # Value function at the theta before the update, i.e. the average reward observed in each queue simulation episode for a fixed policy before the update of theta takes place.
        self.gradV = []             # Gradient of the value function (grad(J) in Sutton) for the theta before the update, that is responsible for the theta update.
        self.G = None               # G(t) for each simulation time step t as estimated by self.learn().

    def reset(self, reset_value_functions=False, reset_counts=False):
        "Resets the simulator"
        # Reset the learner of the value function
        # This is expected to be called after each learning step, i.e. before a new queue simulation is run
        self.agent.getLearnerV().reset(reset_value_functions=reset_value_functions, reset_trajectory=True, reset_counts=reset_counts)
        # Reset the auxiliary attributes of the policy learner (e.g. trajectory, etc.)
        # but NOT the history of theta nor the counts of visited states or actions because
        # these are used to adjust the policy learning rate as necessary, which follows the learning time scale
        # and NOT the simulation time scale.
        self.agent.getLearnerP().reset(reset_value_functions=reset_value_functions, reset_trajectory=False, reset_counts=reset_counts)
        if not reset_value_functions:
            # Set the start value of the value functions to their current estimate
            # Goal: we can use the current estimated values in the next learning step,
            # e.g. use it as a baseline when computing the time difference delta(t).
            self.agent.getLearnerV().setVStart(self.agent.getLearnerV().getV())
            self.agent.getLearnerV().setQStart(self.agent.getLearnerV().getQ())

    def run(self, start_state, seed=None, compute_rmse=False, state_observe=None,
            verbose=False, verbose_period=1,
            plot=False, colormap="seismic", pause=0):
        """Runs a Reinforcement Learning experiment on a queue environment for the given number of time steps.
        No reset of the learning history is done at the beginning of the simulation.
        It is assumed that this reset, when needed, is done by the calling function.

        Parameters:
        start_state: int or list or numpy array
            State at which the queue environment starts.

        seed: None or float, optional
            Seed to use for the random number generator for the simulation.
            If None, the seed is NOT set.
            If 0, the seed stored in the object is used.
            This is useful if this method is part of a set of experiments run and we want to keep
            the seed setting that had been done at the offset of the set of experiments. Otherwise,
            if we set the seed again now, the experiments would have always the same outcome.

        compute_rmse: bool, optional
            Whether to compute the RMSE over states (weighted by their number of visits)
            after each episode. Useful to analyze rate of convergence of the estimates.

        state_observe: int, optional
            A state index whose RMSE should be observed as the episode progresses.

        verbose: bool, optional
            Whether to show the experiment that is being run and the episodes for each experiment.

        verbose_period: int, optional
            Every how many episodes per experiment should be displayed.

        plot: bool, optional
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: str, optional
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.

        pause: float, optional
            Number of seconds to wait before updating the plot that shows the evalution
            of the value function estimates.

        Returns: tuple
        Tuple containing the following elements:
            [TBD]: See the Simulator class for ideas, but it would essentially contain the learning parameters of
            the different learners (e.g. state value function, action-state value function, policy)
            - learnerV: the learner of the value function at the end of the learning process
            - learnerP: the learner of the policy at the end of the learning process.
                The policy can be retrieved from this object by calling learnerP.getPolicy().
            - df_learning: pandas data frame containing two columns: 'theta', 'gradV' containing the history
            of the observed thetas and gradients of the value function responsible for its update.
            - [TODO] number of visits to each state at the end of the simulation.
        """
        # --- Parse input parameters
        if plot:
            # TODO: See the run() method in the Simulator class
            pass

        # Simulation seed
        if seed is not None:
            if seed != 0:
                self.env.set_seed(seed)
            else:
                self.env.set_seed(self.seed)

        if verbose:
            print("Value function at start of experiment: {}".format(self.learnerV.getV()))
        # There are two simulation time scales (both integers and starting at 0):
        # - t: the number of queue state changes (a.k.a. number of queue iterations)
        # - t_learn: the number of learning steps (of the value functions and policies)
        t_learn = -1
        while t_learn < self.dict_nsteps['learn'] - 1:  # -1 because t_learn starts at -1 (think what would happen when dict_nsteps['learn'] = 1: i.e. if we don't put -1, we would run 2 iterations, instead of the requested 1 iteration)
            t_learn += 1
            # Reset the environment and the learners
            # (i.e. start the environment at the same state and prepare it for a fresh new learning step
            # with all learning memory erased)
            job_class = None
            self.env.reset((start_state, job_class))
            self.reset(reset_value_functions=False, reset_counts=False)

            done = False
            if verbose and np.mod(t_learn, verbose_period) == 0:
                print("\n***** Learning step {} of {} running...".format(t_learn + 1, self.dict_nsteps['learn']), end=" ")
                print("\nQueue environment starts at state {}".format(self.env.getState()))
                print("\tState value function:\n\t{}".format(self.learnerV.getV()))
                print("\tTheta parameter of policy: {}".format(self.learnerP.getPolicy().getThetaParameter()))

            # Time step in the queue trajectory (the first time step is t = 0)
            t = -1
            event_prev = None
            while not done:
                # Current state
                state = self.env.getState()

                # Next state
                time, event, job_class_or_server = self.generate_event()
                if event == Event.BIRTH:
                    # The event is an incoming job class
                    # => Increment the discrete time step, set the arriving job class in the queue environment, and apply the acceptance policy
                    # NOTE: Only BIRTH events mark a new discrete time
                    # (this guarantees that the stationary distribution of the discrete-time process coincides with that of the continuous-time process)
                    t += 1

                    # Store the class of the arriving job
                    job_class = job_class_or_server
                    self.env.setJobClass(job_class)

                    # Perform a step on the environment by applying the acceptance policy
                    action_accept_reject, next_state, reward_accept_reject, _ = self.step(t, PolicyTypes.ACCEPT)

                    # Store the gradient of the policy evaluated on the taken action A(t) given the state S(t)
                    # Goal: be able to compute e.g. the average policy gradient when learning at the end of the simulation
                    gradient_for_action = self.learnerP.getPolicy().getGradient(action_accept_reject, state)
                    self.learnerP.record_gradient(state, action_accept_reject, gradient_for_action)

                    # Assign the job to a server if accepted
                    if action_accept_reject == Actions.ACCEPT:
                        # Assign the job just accepted to a server and update the next_state value
                        # (because the next_state from the ACCEPT action has not been updated above
                        # as we need to know to which server the accepted job is assigned in order
                        # to know the next state of the queue environment)
                        _, next_state, reward_assign, _ = self.step(t, PolicyTypes.ASSIGN)
                    else:
                        reward_assign = 0.0

                    action = action_accept_reject
                    reward = reward_accept_reject + reward_assign
                    if action == Actions.ACCEPT and reward != 0.0:
                        print("--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward))
                        raise ValueError("A non-zero reward is not possible when the action is ACCEPT.")

                    # Update the trajectory used in the learning process, where we store:
                    # S(t): state BEFORE an action is taken
                    # A(t): action taken given the state S(t)
                    # R(t): reward received by taking action A(t) and transition to S(t+1)
                    self.update_trajectory(t_learn*(self.dict_nsteps['queue']+1) + t, t, state, action, reward)
                    # Update the average reward (just for information purposes)
                    self.learnerV.updateAverageReward(t, reward)

                    done = self.check_done(t, state, action, reward, gradient_for_action)
                    if done:
                        print("Done at ")
                elif event == Event.DEATH:
                    # The event is a completed service
                    # => Update the state of the queue but do NOT update the discrete-time step
                    server = job_class_or_server
                    queue_state = self.env.getQueueState()
                    assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"
                    next_queue_state = copy.deepcopy(queue_state)
                    next_queue_state[server] -= 1
                    self.env.setState((next_queue_state, None))
                    next_state = self.env.getState()

                    assert self.env.getBufferSizeFromState(next_state) == self.env.getBufferSizeFromState(state) - 1, \
                        "The buffer size after a DEATH decreased by 1: S(t) = {}, S(t+1) = {}" \
                            .format(self.env.getBufferSizeFromState(state), self.env.getBufferSizeFromState(next_state))

                    # We set the action to None because there is no action by the agent when a service is completed
                    action = None
                    # We set the reward to NaN because a reward may happen ONLY when a new job arrives.
                    # If there are rewards due to serviced jobs, we would impact those rewards into the reward
                    # received when a new job arrives (which is not the case at this moment (28-Nov-2021),
                    # i.e. the reward when a job is serviced is 0.
                    reward = np.nan
                    if event_prev == Event.BIRTH or event_prev is None:
                        # Add the state PRIOR to the FIRST DEATH after a BIRTH (or the very first DEATH event
                        # if that is the first event that occurs in the simulation), so that we record the state
                        # to which the system went just after the latest BIRTH (so that the plot that is generated
                        # at the end showing the states of the system at each time step does not raise suspicion
                        # because the buffer size doesn't change between two consecutive time steps --which would be
                        # inconsistent with the fact that a new time step is defined when a new job arrives).
                        self.update_trajectory(t_learn*(self.dict_nsteps['queue']+1) + t, t, state, action, reward)

                if self.debug:
                    print("{} | t={}: event={}, action={} -> state={}, reward={}".format(state, t, event, action,
                                                                                         next_state, reward), end="\n")

                event_prev = event

            if verbose and np.mod(t_learn, verbose_period) == 0:
                print("==> agent ENDS at time t={} at state {} coming from state = {}, action = {}, reward = {}, gradient = {})" \
                    .format(t, self.env.getState(), state, action, reward, gradient_for_action))

            # Learn the value function and the policy
            theta_prev = self.learnerP.getPolicy().getThetaParameter()
            self.learn(t)
            if verbose and np.mod(t_learn, verbose_period) == 0:
                print("\tUpdated value function at the end of the queue simulation: V(s) = {}".format(self.learnerV.getV()))
                print("\tObserved average reward = {}".format(self.learnerP.getAverageRewardUnderPolicy()))
                print("\tUpdated theta parameter of policy after learning: theta = {} -> {}".format(theta_prev, self.learnerP.getPolicy().getThetaParameter()))

        if plot:
            # TODO
            pass

        df_learning = pd.DataFrame({'theta': self.thetas, 'theta_next': self.thetas_updated, 'V': self.V, 'gradV': self.gradV})

        return self.learnerV, self.learnerP, df_learning
        # , self.learnerV.getStateCounts(), \
        # {  # Value of alpha for each state at the end of the LAST t_learn run
        #    'average alpha': self.learnerV.getAverageLearningRates(),
        # (Average) alpha by t_learn (averaged over visited states in the t_learn)
        #    'alphas used': self.learnerV.getLearningRates()
        # }

    def generate_event(self):
        """
        Generates the next event for the queue system which can be either:
        - an incoming job (of a given class)
        - a completed service

        Return: tuple
        Tuple containing the following elements:
        - the time of the event
        - the event type (Event.BIRTH or Event.DEATH)
        - an index that represents either:
            - the job class, if the generated event is an arriving job
            - the server, if the generated event is a completed service
        """
        # Boolean indices defining the rates to consider in the generation of the next event
        # Servers that are EMPTY do not comply with this condition
        n_job_classes = self.env.getNumJobClasses()
        is_valid_rate = np.r_[
            [True] * n_job_classes,
            [True if self.env.getQueueState()[s] > 0 else False for s in range(self.env.getNumServers())]
        ]

        rates = np.r_[self.env.getJobClassRates(), self.env.getServiceRates()]
        valid_rates = rates
        valid_rates[~is_valid_rate] = np.nan

        # Generate the event time and the index on the rates array to which it is associated
        event_time, event_idx = generate_min_exponential_time(valid_rates)

        # Define whether the generated event is an incoming job class or a completed service
        if event_idx < n_job_classes:
            # The event is an incoming job class
            job_class = event_idx
            return event_time, Event.BIRTH, job_class
        else:
            # The event is a completed service event
            idx_server = event_idx - n_job_classes
            return event_time, Event.DEATH, idx_server

    def step(self, t, policy_type):
        """
        Arguments:
        t: int
            Current queue transition associated to the current step.

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
        action, observation, reward, info = self.agent.act(policy_type)

        return action, observation, reward, info

    def update_trajectory(self, t_total, t_sim, state, action, reward):
        """

        Arguments:
        t_total: int
            Total time, used to store the trajectory for the policy learning.
            Normally this would be the total simulation time computed from the learning steps.

        t_sim: int
            Simulation time, used to store the trajectory for the value function learning.
        """
        self.learnerV.update_trajectory(t_sim, state, action, reward)
        # DM-2021/11/28: This assertion is no longer true because we are now storing in the trajectory ALSO the states
        # occurring just before the FIRST DEATH event happening after a BIRTH event (so that we can show it in the
        # trajectory plot we show at the end of the simulation and thus avoid the suspicion that something is wrong
        # when we observe no change in the buffer size from one time step to the next --given that a new time step
        # is created ONLY when a new job arrives (i.e. a BIRTH event occurs)
        #assert action is not None, "The action is not None when learning the policy"
        self.learnerP.update_trajectory(t_total, state, action, reward)

    def learn(self, t):
        """
        Learns the value functions and the policy at time t

        Arguments:
        t: int
            Time (iteration) in the queue simulation time scale at which learning takes place.
            In Monte-Carlo learning, this time would be the end of the queue simulation.
        """
        # TODO: (2021/11/03) Think if we can separate the learning of the value function and return G(t) from the learning of the policy (theta)
        # Goal: avoid computing G(t) twice, one at learnerV.learn() and one at learnerP.learn()
        # Note HOWEVER that the G(t) learned in learnerV is NOT the same as the G(t) learned in leanerP, because the latter
        # uses a corrected computation of G(t), i.e. sum_{s>=t}{ r(s) - rho }, where rho is the average reward
        # observed under a policy... well, the calculation of rho could be embedded in learnerV as well, as in learnerV
        # we assume that the policy is fixed.
        # For now I am commenting out learnerV.learn() because we don't actually use the value of G(t) computed there
        # when updating the policy in learnerP.learn().
        self.learnerV.learn(t)  # UNCOMMENTED ON SAT, 27-NOV-2021
        #theta_prev, theta, V, gradV, G = self.learnerP.learn(t)
        theta_prev, theta, V, gradV, G = self.learnerP.learn_TR(t)
        self.thetas += [theta_prev]
        self.thetas_updated += [theta]
        self.V += [V]
        self.gradV += [gradV]
        self.G = G

        self.learnerP.update_learning_time()
        print("*** Learning-time updated: time = {} ***".format(self.learnerP.getLearningTime()))
        self.learnerP.update_alpha_by_learning_episode()

    def check_done(self, t, state, action, reward, gradient):
        """
        Checks whether the simulation is done

        t: int
            Current queue simulation time.

        state: Environment dependent
            S(t): state of the environment at time t, BEFORE the action is taken.

        action: Environment dependent
            A(t): action received by the environment at time t.

        reward: float
            R(t+1): reward yielded by the environment after taking action A(t) at state S(t).

        gradient: float
            Gradient of the policy at time t, i.e. the value of the policy gradient for A(t) given S(t).
            This value COULD be used to determine whether the simulation is done, i.e. if the gradient is not zero
            then we may assume that the simulation is done because we would have a step that allows us to learn
            theta at that time.

        Return: bool
            Whether the queue simulation is done because the maximum number of iterations has been reached.
        """
        if t < self.dict_nsteps['queue']:  # and gradient == 0.0:
            done = False
        else:
            done = True

        return done

    # ------ GETTERS ------#
    def getJobRatesByServer(self):
        return self.job_rates_by_server


if __name__ == "__main__":
    import runpy

    runpy.run_path("../../setup.py")

    if False:
        #--- Test the Simulator class ---#
        # NOTE: (2021/10/19) This test works
        # from Python.lib import environments, agents
        from environments import gridworlds
        from Python.lib.agents.policies import random_walks
        from agents.learners.episodic.discrete import mc, td

        # Plotting setup
        colormap = cm.get_cmap("jet")
        max_alpha = 1
        max_rmse = 0.5
        #fig, (ax_full, ax_scaled, ax_rmse_by_episode) = plt.subplots(1,3)
        fig, (ax_full, ax_scaled) = plt.subplots(1,2)

        # The environment
        env = gridworlds.EnvGridworld1D(length=21)

        # Possible policies and learners for agents
        pol_rw = random_walks.PolRandomWalkDiscrete(env)
        lea_td = td.LeaTDLambda(env)
        lea_mc = mc.LeaMCLambda(env)

        # Policy and learner to be used in the simulation
        policy = pol_rw
        learner = lea_td

        # Simulation setup
        seed = 1717
        nexperiments = 1
        nepisodes = 10
        start = None
        useGrid = False
        verbose = True
        debug = False

        # Define hyperparameter values
        gamma = 0.8
        if useGrid:
            n_lambdas = 10
            n_alphas = 10
            lambdas = np.linspace(0, 1, n_lambdas)
            alphas = np.linspace(0.1, 0.7, n_alphas)
        else:
            lambdas = [0, 0.4, 0.8, 0.9, 0.95, 0.99, 1]
            alphas = np.linspace(0.1, 0.9, 8)
            lambdas = [0, 0.4, 0.7, 0.8, 0.9]
            #alphas = [0.41]
            n_lambdas = len(lambdas)
            n_alphas = len(alphas)
        n_simul = n_lambdas*n_alphas

        # List of dictionaries, each containing the characteristic of each parameterization considered
        results_list = []
        legend_label = []
        # Average RMSE obtained at the LAST episode run for each parameter set and their standard error
        rmse_mean_values = np.nan*np.zeros((n_lambdas, n_alphas))
        rmse_se_values = np.nan*np.zeros((n_lambdas, n_alphas))
        # Average RMSE over the episodes run for each parameter set
        rmse_episodes_mean = np.nan*np.zeros((n_lambdas, n_alphas))
        rmse_episodes_se = np.nan*np.zeros((n_lambdas, n_alphas))
        # RMSE over the episodes averaged over ALL parameter set runs
        rmse_episodes_values = np.zeros(nepisodes)
        idx_simul = -1
        for idx_lmbda, lmbda in enumerate(lambdas):
            rmse_mean_lambda = []
            rmse_se_lambda = []
            rmse_episodes_mean_lambda = []
            rmse_episodes_se_lambda = []
            for alpha in alphas:
                idx_simul += 1
                if verbose:
                    print("\nParameter set {} of {}: lambda = {:.2g}, alpha = {:.2g}" \
                          .format(idx_simul, n_simul, lmbda, alpha))

                # Reset learner and agent (i.e. erase all memory from a previous run!)
                learner.setParams(alpha=alpha, gamma=gamma, lmbda=lmbda)
                learner.reset(reset_episode=True, reset_value_functions=True)
                agent = GenericAgent(pol_rw,
                                     learner)
                # NOTE: Setting the seed here implies that each set of experiments
                # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
                # of visited states and actions.
                # This is DESIRED --as opposed of having different state-action outcomes for different
                # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
                # VERIFIED BY RUNNING IN DEBUG MODE!
                sim = Simulator(env, agent, seed=seed, debug=debug)

                # Run the simulation and store the results
                N_mean, rmse_mean, rmse_se, rmse_episodes, _, learning_info = \
                                    sim.simulate(nexperiments=nexperiments,
                                                 nepisodes=nepisodes,
                                                 start=start,
                                                 verbose=verbose)
                results_list += [{'lmbda': lmbda,
                                  'alpha': alpha,
                                  'rmse': rmse_mean,
                                  'SE': rmse_se
                                 }]
                rmse_mean_lambda += [rmse_mean]
                rmse_se_lambda += [rmse_se]
                rmse_episodes_mean_lambda += [np.mean(rmse_episodes)]
                rmse_episodes_se_lambda += [np.std(rmse_episodes) / np.sqrt(nepisodes)]
                rmse_episodes_values += rmse_episodes

                if verbose:
                    print("\tRMSE = {:.3g} ({:.3g})".format(rmse_mean, rmse_se))

            rmse_mean_values[idx_lmbda] = np.array(rmse_mean_lambda)
            rmse_se_values[idx_lmbda] = np.array(rmse_se_lambda)
            rmse_episodes_mean[idx_lmbda] = np.array(rmse_episodes_mean_lambda)
            rmse_episodes_se[idx_lmbda] = np.array(rmse_episodes_se_lambda)

            # Plot the average RMSE for the current lambda as a function of alpha
            #rmse2plot = rmse_mean_lambda
            #rmse2plot_error = rmse_se_lambda
            #ylabel = "Average RMSE over all {} states, at the end of episode {}, averaged over {} experiments".format(env.getNumStates(), nepisodes, nexperiments)
            rmse2plot = rmse_episodes_mean_lambda
            rmse2plot_error = rmse_episodes_se_lambda
            ylabel = "Average RMSE over all {} states, first {} episodes, and {} experiments".format(env.getNumStates(), nepisodes, nexperiments)

            # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
            color = colormap( 1 - idx_lmbda / np.max((1, n_lambdas-1)) )
            ax_full.plot(alphas, rmse2plot, '.', color=color)
            ax_full.errorbar(alphas, rmse2plot, yerr=rmse2plot_error, capsize=4, color=color)
            legend_label += ["lam={:.2g}".format(lmbda)]

        # Average RMSE by episode for convergence analysis
        rmse_episodes_values /= n_simul

        # JUST ONE FINAL RUN
        #learner.setParams(alpha=0.3, gamma=gamma, lmbda=1)
        #agent = GenericAgent(pol_rw,
        #                     learner)
        #sim = Simulator(env, agent)
        #rmse_mean, rmse_se = sim.simulate(nexperiments=nexperiments, nepisodes=nepisodes, verbose=verbose)
        #if verbose:
        #    print("\tRMSE = {:.3g} ({:.3g})".format(rmse_mean, rmse_se))

        # Scaled plot (for comparison purposes)
        for idx_lmbda, lmbda in enumerate(lambdas):
            #rmse2plot = rmse_mean_values[idx_lmbda]
            #rmse2plot_error = rmse_se_values[idx_lmbda]
            #ylabel = "Average RMSE over all {} states, at the end of episode {}, averaged over {} experiments".format(env.getNumStates(), nepisodes, nexperiments)
            rmse2plot = rmse_episodes_mean[idx_lmbda]
            rmse2plot_error = rmse_episodes_se[idx_lmbda]
            ylabel = "Average RMSE over all {} states, first {} episodes, and {} experiments".format(env.getNumStates(), nepisodes, nexperiments)
            color = colormap( 1 - idx_lmbda / np.max((1, n_lambdas-1)) )
            ax_scaled.plot(alphas, rmse2plot, '.-', color=color)
            ax_scaled.errorbar(alphas, rmse2plot, yerr=rmse2plot_error, capsize=4, color=color)
            ax_scaled.set_xlim((0, max_alpha))
            ax_scaled.set_ylim((0, max_rmse))

        # Episodic RMSE
        #ax_rmse_by_episode.plot(np.arange(nepisodes), rmse_episodes_values, color="black")
        #ax_rmse_by_episode.set_ylim((0, max_rmse))
        #ax_rmse_by_episode.set_xlabel("Episode")
        #ax_rmse_by_episode.set_ylabel("RMSE")
        #ax_rmse_by_episode.set_title("Average RMSE by episode over ALL experiments")

        plt.figlegend(legend_label)
        fig.suptitle("{}: gamma = {:.2g}, #experiments = {}, #episodes = {}"\
                     .format(learner.__class__, gamma, nexperiments, nepisodes))

    if True:
        # --- Test the SimulatorQueue class ---#
        from agents.queues import AgeQueue, LearnerTypes
        from environments.queues import EnvQueueSingleBufferWithJobClasses
        from agents.policies.parameterized import PolQueueTwoActionsLinearStep, PolQueueTwoActionsLogit
        from agents.learners.continuing.mc import LeaMC
        from agents.learners.policies import LeaPolicyGradient

        capacity = np.Inf
        nservers = 1
        job_class_rates = [0.7]
        service_rates = [1.0]
        queue = QueueMM(job_class_rates, service_rates, nservers, capacity)
        env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobRejection_ExponentialCost, None)

        # Acceptance policy definition
        theta_start = 1.3  # 1.3, 11.3  # IMPORTANT: This value should NOT be integer, o.w. the policy gradient will always be 0 regardless of the state at which blocking occurs
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue_mm, theta_start), PolicyTypes.ASSIGN: None})
        #policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLogit(env_queue_mm, theta_start, beta=1.0), PolicyTypes.ASSIGN: None})

        # Simulator on a given number of iterations for the queue simulation and a given number of iterations to learn
        t_sim = 5000 #10
        t_learn = 50 #1

        # Learners definition
        fixed_window = False
        alpha_start = 1.0 / t_sim  # Use `/ t_sim` when using update of theta at each simulation step (i.e. LeaPolicyGradient.learn_TR() is called instead of LeaPolicyGradient.learn())
        adjust_alpha = False
        min_time_to_update_alpha = 40
        alpha_min = 0.001
        gamma = 1.0
        learnerV = LeaMC(env_queue_mm, gamma=gamma)
        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(env_queue_mm, policies[PolicyTypes.ACCEPT], learnerV,
                                                           alpha=alpha_start, adjust_alpha=adjust_alpha,
                                                           min_time_to_update_alpha=min_time_to_update_alpha,
                                                           alpha_min=alpha_min, fixed_window=fixed_window)})
        agent_gradient_mc = AgeQueue(env_queue_mm, policies, learners)

        dict_nsteps = dict({'queue': t_sim, 'learn': t_learn})
        start_state = [0]
        simul = SimulatorQueue(env_queue_mm, agent_gradient_mc, dict_nsteps, debug=False)
        learnerV, learnerP, df_learning = simul.run(start_state, seed=1717, verbose=True)

        # Save the estimation of G(t) for the last learning step to a file
        #file_results_G = "G.csv"
        #pd.DataFrame({'G': simul.G}).to_csv(file_results_G)

        #-- Plot theta and the gradient of the value function
        SET_YLIM = True

        # Estimated value function
        ax, line_est = plotting.plot_colormap(df_learning['theta'], -df_learning['V'], cmap_name="Blues")

        # True value function
        # Block size for each theta, defined by the fact that K-1 is between theta and theta+1 => K = ceiling(theta+1)
        Ks = [np.int(np.ceil(np.squeeze(t)+1)) for t in df_learning['theta']]
        rho = job_class_rates[0] / service_rates[0]
        # Blocking probability = Pr(K)
        p_stationary = [stationary_distribution_birth_death_process(1, K, [rho])[1] for K in Ks]
        pblock_K = np.array([p[-1] for p in p_stationary])
        pblock_Km1 = np.array([p[-2] for p in p_stationary])
        # Blocking probability adjusted for different jump rates between K-1 and K (affected by the non-deterministic probability of blocking at K-1)
        pblock_K_adj = np.squeeze([pK * (1 - (K-1-theta)) for K, theta, pK in zip(Ks, df_learning['theta'], pblock_K)])
        pblock_Km1_adj = pblock_Km1 #np.squeeze([pKm1 + pK - pK_adj for pKm1, pK, pK_adj in zip(pblock_Km1, pblock_K, pblock_K_adj)])
        #assert np.allclose(pblock_K + pblock_Km1, pblock_K_adj + pblock_Km1_adj)
        # True value function: expected cost at K which is the buffer size where blocking most likely occurs...
        # (in fact, if theta is say 3.1, the probability of blocking at 4 (= K-1) is small and most blocking
        # will occur at K; if theta is 3.9, the probability of blocking at 4 (= K-1)
        # i.e. we compute at K-1 and NOT at K because we want to compare the true value function
        # with the *estimated* value function when the policy starts blocking at buffer size = theta
        #Vtrue = np.array([rewardOnJobRejection_ExponentialCost(env_queue_mm, (K, None), Actions.REJECT, (K, None)) * pK for K, pK in zip(Ks, pblock_K)])

        # ACTUAL true value function, which takes into account the probability of blocking at K-1 as well, where the policy is non-deterministic (for non-integer theta)
        # The problem with this approach is that the stationary distribution of the chain is NOT the same as with chain
        # where rejection ONLY occurs at s=K... in fact, the transition probabilities to s=K and to s=K-1 when the
        # initial state is s=K-1 are affected by the non-deterministic probability of blocking when s=K-1...
        # Qualitatively, the stationary probability of K would be reduced and the stationary probability of K-1 would be
        # increased by the same amount.
        Vtrue = np.array([rewardOnJobRejection_ExponentialCost(env_queue_mm, (K, None), Actions.REJECT, (K, None)) * pK +
                          rewardOnJobRejection_ExponentialCost(env_queue_mm, (K-1, None), Actions.REJECT, (K-1, None)) * (K - 1 - theta) * pKm1
                            for K, theta, pK, pKm1 in zip(Ks, df_learning['theta'], pblock_K_adj, pblock_Km1_adj)])

        # True grad(V)
        # Ref: my hand-written notes in Letter-size block of paper with my notes on the general environment - agent setup
        gradVtrue = [-rewardOnJobRejection_ExponentialCost(env_queue_mm, (K-1, None), Actions.REJECT, (K-1, None)) * pKm1 for K, pKm1 in zip(Ks, pblock_Km1)]

        ord = np.argsort(Ks)
        # NOTE that we plot the true value function at K-1 (not at K) because K-1 is the value that is closest to theta
        # and we are plotting the *estimated* value function vs. theta (NOT vs. K).
        #line_true, = ax.plot([Ks[o]-1 for o in ord], [-Vtrue[o] for o in ord], 'g.-', linewidth=5, markersize=20)
        line_true, = ax.plot(df_learning['theta'], -Vtrue, 'gx-')  # Use when computing the ACTUAL true Value function V, which also depends on theta!
        ax.set_yscale('log')
        #ax.set_ylim((0, 10))
        ax.set_xlabel('theta (for estimated functions) / K-1 for true value function')
        ax.set_ylabel('Value function V (cost)')
        ax.legend([line_est, line_true], ['Estimated V', 'True V'], loc='upper left')
        ax2 = ax.twinx()
        ax2, line_grad = plotting.plot_colormap(df_learning['theta'], -df_learning['gradV'], cmap_name="Reds", ax=ax2)
        line_gradtrue, = ax2.plot([Ks[o]-1 for o in ord], [-gradVtrue[o] for o in ord], 'k.-', linewidth=3, markersize=12)
        ax2.axhline(0, color="lightgray")
        ax2.set_ylabel('grad(V)')
        if SET_YLIM:
            ax2.set_ylim((-5,5))        # Note: grad(V) is expected to be -1 or +1...
        ax2.legend([line_grad, line_gradtrue], ['grad(V)', 'True grad(V)'], loc='upper right')
        plt.title("Value function and its gradient as a function of theta and K. " +
                    "Optimum K = {}, Theta start = {}, t_sim = {:.0f}".format(COST_EXP_BUFFER_SIZE_REF, theta_start, t_sim))

        # grad(V) vs. V
        plt.figure()
        plt.plot(-df_learning['V'], -df_learning['gradV'], 'k.')
        ax = plt.gca()
        ax.axhline(0, color="lightgray")
        ax.axvline(0, color="lightgray")
        ax.set_xscale('log')
        #ax.set_xlim((-1, 1))
        if SET_YLIM:
            ax.set_ylim((-1, 1))
        ax.set_xlabel('Value function V (cost)')
        ax.set_ylabel('grad(V)')

        # Plot evolution of theta
        # NOTE: (2021/11/27) I verified that the values of learnerP.getRewards() for the last learning step are the same to those for learnerV.getRewards() (which only stores the values for the LAST learning step)
        plt.figure()
        # Times at which the trajectory of states is recorded
        times = learnerP.getTimes()
        times_unique = np.unique(times)
        assert len(times_unique) == len(learnerP.getPolicy().getThetas())
        plt.plot(times_unique, learnerP.getPolicy().getThetas(), 'b.-')
        ax = plt.gca()
        # Add vertical lines signalling the BEGINNING of each queue simulation
        times_sim_starts = range(0, t_learn*(t_sim+1), t_sim+1)
        for t in times_sim_starts:
            ax.axvline(t, color='lightgray', linestyle='dashed')

        # Buffer sizes
        buffer_sizes = [env_queue_mm.getBufferSizeFromState(s) for s in learnerP.getStates()]
        ax.plot(times, buffer_sizes, 'g.', markersize=3)
        # Mark the start of each queue simulation
        # DM-2021/11/28: No longer feasible (or easy to do) because the states are recorded twice for the same time step (namely at the first DEATH after a BIRTH event)
        #ax.plot(times_sim_starts, [buffer_sizes[t] for t in times_sim_starts], 'gx')
        ax.set_xlabel("time step")
        ax.set_ylabel("theta")

        # Secondary plot showing the rewards received
        ax2 = ax.twinx()
        ax2.plot(times, -np.array(learnerP.getRewards()), 'r-', alpha=0.3)
        # Highlight with points the non-zero rewards
        ax2.plot(times, [-r if r != 0.0 else None for r in learnerP.getRewards()], 'r.')
        ax2.set_ylabel("Reward")
        ax2.set_yscale('log')
        plt.title("Optimum Theta = {}, Theta start = {}, t_sim = {:.0f}, fixed_window={}".format(COST_EXP_BUFFER_SIZE_REF, theta_start, t_sim, fixed_window))
