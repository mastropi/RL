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

if __name__ == "__main__":
    import runpy
    runpy.run_path('../../setup.py')

import copy
import warnings
from enum import Enum, unique

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm  # cm is for colormaps (e.g. cm.get_cmap())
from timeit import default_timer as timer

from agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
from agents.learners.continuing.fv import LeaFV
from agents.queues import PolicyTypes
from environments.queues import rewardOnJobRejection_ExponentialCost, Actions, COST_EXP_BUFFER_SIZE_REF
from estimators import estimate_blocking_fv_separately, estimate_proba_survival_and_expected_absorption_time_mc

from utils.basic import merge_values_in_time, index_linear2multi, measure_exec_time
from utils.computing import rmse, compute_job_rates_by_server, generate_min_exponential_time, stationary_distribution_birth_death_process
import utils.plotting as plotting
from queues import QueueMM, Event

@unique
class LearningMethod(Enum):
    MC = 1
    FV = 2


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

    def reset(self, reset_episode=True, reset_value_functions=True):
        "Resets the simulator"
        # Copy of Initial State Distribution of environment in case we need to change it
        self._isd_orig = None

        # Reset the learner to the first episode state
        self.agent.getLearner().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)

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
            default: False

        verbose_period: int, optional
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

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
            default: False

        verbose_period: int, optional
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

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

    N: int
        Number of queues to simulate.
        default: 1
    """

    def __init__(self, env, agent, dict_nsteps, N=1, seed=None, debug=False):
        super().__init__(env, agent, seed, debug)
        self.dict_nsteps = dict_nsteps
        # Attributes used during Fleming-Viot simulation (each env in envs is one of the N particles)
        self.N = N
        # Note: it's important that the first environment stored in self.envs be self.env (and NOT a copy)
        # because in the process we sometimes refer to it as self.envs[0] (for instance when resetting the environments)
        # or as self.env (for instance when estimating Q(K-1, a) via Monte-Carlo on one particle (both in MC and FV modes)
        self.envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(self.N)]

        # Job-rates by server (assuming jobs are pre-assigned to servers)
        # TODO: (2021/12/11) Commented following line because these rates are not used for now
        #self.job_rates_by_server = compute_job_rates_by_server(self.env.getJobClassRates(),
        #                                                       self.env.getNumServers(),
        #                                                       self.agent.getAssignmentPolicy().getProbabilisticMap())

        # Learners for the value function and the policy
        if agent is not None:
            self.learnerV = self.agent.getLearnerV()
            self.learnerP = self.agent.getLearnerP()
        else:
            self.learnerV = None
            self.learnerP = None

        # Storage of learning history
        self.alphas = []            # Learning rates used in the learning of the policy when updating from theta to theta_next (i.e. from thetas -> thetas_updated)
        self.thetas = []            # List of theta values before the update by the policy learning step
        self.thetas_updated = []    # List of theta values AFTER the update by the policy learning step.
        self.proba_stationary = []  # List of the estimates of the stationary probability of K-1 at theta
        self.Q_diff = []            # List of the Q(K-1,a=1) - Q(K-1,a=0) values leading to the udpate of theta
        self.V = []                 # *Normally*, the value function at the theta before the update,
                                    # i.e. the average reward rho observed in each queue simulation episode for a fixed policy before the update of theta takes place.
                                    # HOWEVER, it is NOT always the case that this is the average reward, see the self.learn() method for more details.
        self.gradV = []             # Gradient of the value function (grad(J) in Sutton) for the theta before the update, that is responsible for the theta update.
        self.G = None               # G(t) for each simulation time step t as estimated by self.learn().

    # IMPORTANT: This method overrides the method of the superclass! (so when reset() is called from the super()
    # object, actually THIS reset() method defined here is called!
    def reset(self, state=None, reset_value_functions=False, reset_counts=False):
        "Resets the environments used in the simulation to the given state and the learners for a new learning process"
        # Reset the environments (particles) used in the simulation
        if state is not None:
            for env in self.envs:
                env.reset(state)

        if self.agent is not None:
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

    def run(self, dict_params_simul: dict, dict_params_info: dict={'plot': False, 'log': False},
            start_state=None,
            seed=None, state_observe=None,
            verbose=False, verbose_period=1,
            colormap="seismic", pause=0):
        """
        Runs a Reinforcement Learning experiment on a queue environment for the given number of time steps.
        No reset of the learning history is done at the beginning of the simulation.
        It is assumed that this reset, when needed, is done by the calling function.

        Parameters:
        dict_params_simul: dict
            Dictionary containing the simulation parameters as follows:
            - 'nparticles' (used only in the FV simulation)
            - 'nmeantimes' (used only in the FV simulation)
            - 'buffer_size_activation' (this value should NOT be set as it is set by the simulation process)
            - 'seed' (this value should NOT be given as it is set by the simulation process)

        dict_params_info: (opt) dict --> only used when the learner of the value functions `learnerV` is LeaFV
            Dictionary containing general info parameters as follows:
            - 'plot': whether to create plots
            - 'log': whether to show messages
            default: None

         start_state: (opt) int or list or numpy array
            State at which the queue environment starts.
            default: None, in which case the start state is defined by the simulation process, based on the learning requirements

        seed: (opt) None or float
            Seed to use for the random number generator for the simulation.
            If None, the seed is NOT set.
            If 0, the seed stored in the object is used.
            This is useful if this method is part of a set of experiments run and we want to keep
            the seed setting that had been done at the offset of the set of experiments. Otherwise,
            if we set the seed again now, the experiments would have always the same outcome.
            default: None

        state_observe: (opt) int
            A state index whose RMSE should be observed as the episode progresses.
            default: None

        verbose: (opt) bool
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        plot: (opt) bool
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: (opt) str
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.

        pause: (opt) float
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
        if dict_params_info['plot']:
            # TODO: See the run() method in the Simulator class
            pass

        # Simulation seed
        if seed is not None:
            if seed != 0:
                self.env.set_seed(seed)
            else:
                self.env.set_seed(self.seed)
        # --- Parse input parameters

        if verbose:
            print("Value function at start of experiment: {}".format(self.learnerV.getV()))
        # There are two simulation time scales (both integers and starting at 0):
        # - t: the number of queue state changes (a.k.a. number of queue iterations)
        # - t_learn: the number of learning steps (of the value functions and policies)
        t_learn = -1
        while t_learn < self.dict_nsteps['learn'] - 1:  # -1 because t_learn starts at -1 (think what would happen when dict_nsteps['learn'] = 1: i.e. if we don't put -1, we would run 2 iterations, instead of the requested 1 iteration)
            t_learn += 1
            # Reset the learners
            self.reset(reset_value_functions=False, reset_counts=False)

            if verbose and np.mod(t_learn, verbose_period) == 0:
                print("\n************************ Learning step {} of {} running ****************************" \
                      .format(t_learn + 1, self.dict_nsteps['learn']))
                print("Theta parameter of policy: {}".format(self.learnerP.getPolicy().getThetaParameter()))

            # Buffer size of sure blocking which is used to define the buffer size for activation and the start state for the simulations run
            K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()
            dict_params_simul['buffer_size_activation'] = np.max([1, int(K / 4)])
            if isinstance(self.learnerV, LeaFV):
                # TODO: (2021/12/16) SHOULD WE MOVE ALL this learning process of the average reward and stationary probabilities to the learn() method of the LeaFV class?
                # At this point I don't think we should do this, because the learn() method in the GenericLearner subclasses
                # learns the value function from the RESULTS of a SIMULATION ALREADY CARRIED OUT.
                # (see for instance the LeaMC.learn() method)
                # If we do the above change, we should ALSO change the learn() method of LeaMC to perform the simulation
                # of the system itself... and I don't think the learn() method should be responsible for the simulation
                # because simulating is a quite computing intensive task!!

                #-- Estimation of the survival probability P(T>t) and the expected absorption cycle time, E(T_A) by MC
                # We set the activation buffer size as a specific fraction of the buffer size K of sure blocking.
                # This fraction is chosen based on our initial experiments estimating the blocking probability
                # where we obtain the smallest MSE for the estimated blocking probability (between 0.2K and 0.3K).
                print("Running Monte-Carlo simulation on one particle starting at an absorption state with buffer size = {}..." \
                      .format(dict_params_simul['buffer_size_activation'] - 1))
                dict_params_simul['seed'] = self.env.get_seed() + t_learn
                est_mc, expected_absorption_time, n_absorption_time_observations, \
                    proba_surv, n_survival_curve_observations = \
                        estimate_proba_survival_and_expected_absorption_time_mc(self.env, self.agent, dict_params_simul, dict_params_info)
                print("\n*** RESULTS OF MC ESTIMATION OF P(T>t) and E(T) ***")
                #print("P(T>t):\n{}".format(proba_surv))
                print("E(T) = {:.1f}, Max observed survival time = {:.1f}".format(expected_absorption_time, proba_surv['t'].iloc[-1]))

                #-- FLeming-Viot simulation to compute the empirical distribution and then estimate the average reward
                # The empirical distribution Phi(t, bs) estimates the conditional probability of buffer sizes bs
                # for which the acceptance policy is > 0
                assert self.N > 1, "The simulation system has more than one particle in Fleming-Viot mode ({})".format(self.N)
                print("Running Fleming-Viot simulation on {} particles and absorption size = {}..." \
                      .format(self.N, dict_params_simul['buffer_size_activation'] - 1))
                t, event_times, phi, probas_stationary, average_reward = \
                    self.run_simulation_fv( t_learn,
                                            dict_params_simul['buffer_size_activation'] - 1,   # We pass the absorption size to this function
                                            proba_surv, expected_absorption_time,
                                            verbose=verbose, verbose_period=verbose_period)
                print("\n*** RESULTS OF FLEMING-VIOT SIMULATION ***")
                #print("Phi(t):\n{}".format(phi))  # 2021/12/10: phi is WRONG as it always increases with time!!
                print("P(K-1), P(K): {}".format(probas_stationary))
                print("rho = {}".format(average_reward))
                self.learnerV.setAverageReward(average_reward)
                self.learnerV.setProbasStationary(probas_stationary)
            else:
                # Monte-Carlo learning is the default
                assert self.N == 1, "The simulation system has only one particle in Monte-Carlo mode ({})".format(self.N)
                # Smart selection of the start state as one having buffer size = K-1 so that we can better estimate Pr(K-1)
                #start_state = self.choose_state_for_buffer_size(self.env, K)
                # Selection of the start state as one having buffer size = J, in order to have a fair(?) comparison with the FV method
                start_state = self.choose_state_for_buffer_size(self.env, dict_params_simul['buffer_size_activation'])
                t = self.run_simulation_mc(t_learn, start_state, verbose=verbose, verbose_period=verbose_period)
                probas_stationary = self.estimate_stationary_probability_mc()

            if probas_stationary[K-1] > 0.0:
                # When the parameterized policy is a linear step linear function with only one buffer size where its
                # derivative is non-zero, only the DIFFERENCE of two state-action values impacts the gradient, namely:
                # Q(K-1, a=1) - Q(K-1, a=0)
                print("\nEstimating the difference of the state-action values...")
                N = 100; t_sim_max = 250
                K, Q0, Q1, n, max_t = self.estimate_Q_values_until_mixing(t_learn, t_sim_max=t_sim_max, N=N, verbose=verbose, verbose_period=verbose_period)
                #K, Q0, Q1, n, max_t = self.estimate_Q_values_until_stationarity(t_learn, t_sim_max=50, N=N, verbose=verbose, verbose_period=verbose_period)
                print("--> Estimated state-action values on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:\nQ(K-1={}, a=1) = {}\nQ(K-1={}, a=0) = {}\nQ_diff = Q(K-1,1) - Q(K-1,0) = {}" \
                      .format(n, N, max_t, t_sim_max, K - 1, Q1, K - 1, Q0, Q1 - Q0))
            else:
                Q0 = 0.0
                Q1 = 0.0
                print("Estimation of Q_diff skipped because the estimated stationary probability Pr(K-1) = 0.")
            print("--> Estimated stationary probability: Pr(K-1={}) = {}".format(K-1, probas_stationary[K - 1]))

            # Learn the value function and the policy
            theta_prev = self.learnerP.getPolicy().getThetaParameter()

            # Use this when using REINFORCE to learn theta
            #self.learn(self.agent, t)
            # Use this when estimating the theoretical grad(V)
            assert K-1 in probas_stationary.keys(), "The buffer size K-1={} is among the keys of the dictionary containing the stationary probabilities estimation ({})".format(K-1, probas_stationary)
            self.learn(self.agent, t, proba_stationary=probas_stationary[K-1], Q_values=[Q0, Q1])

            if verbose and np.mod(t_learn, verbose_period) == 0:
                print("\tUpdated value function at the end of the queue simulation: average reward V = {}".format(self.learnerV.getV()))
                print("\tSame observed average reward (computed from Policy learner) = {}".format(self.learnerP.getAverageRewardUnderPolicy()))
                print("\tUpdated theta parameter of policy after learning: theta = {} -> {}".format(theta_prev, self.learnerP.getPolicy().getThetaParameter()))

        if dict_params_info['plot']:
            # TODO
            pass

        df_learning = pd.DataFrame.from_items([('theta', self.thetas),
                                               ('theta_next', self.thetas_updated),
                                               ('Pr(K-1)', self.proba_stationary),
                                               ('Q_diff', self.Q_diff),
                                               ('alpha', self.alphas),
                                               ('V', self.V),
                                               ('gradV', self.gradV)
                                                ])

        return self.learnerV, self.learnerP, df_learning

    @measure_exec_time
    def run_simulation_mc(self, t_learn, start_state, agent=None, t_sim_max=None, verbose=False, verbose_period=1):
        """
        Runs the simulation using Monte-Carlo

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        start_state: int or list or numpy array
            State at which the queue environment starts for the simulation.

        agent: (opt) Agent
            Agent object that is responsible of performing the actions on the environment and learning from them.
            default: None, in which case the agent defined in the object is used

        t_sim_max: (opt) int
            Maximum discrete simulation time allowed for the simulation. This is defined ONLY by job arrivals,
            NOT by service events.
            default: None, in which case the number of steps defined in the self.dict_nsteps['queue'] entry is used

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: int
        The last time step of the simulation process.
        """
        #-- Parse input parameters
        if agent is None:
            agent = self.agent
        if t_sim_max is None:
            tmax = self.dict_nsteps['queue']
        else:
            tmax = t_sim_max

        # Set the start state of the environment to the given start state
        job_class = None
        self.env.setState((start_state, job_class))
        print("MC simulation: The queue environments starts at state {}".format(self.env.getState()))

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = -1
        event_prev = None
        while not done:
            # Current state
            state = self.env.getState()

            # Next state
            # Note: this function takes care of NOT generating a service event when a server is empty.
            time, event, job_class_or_server, _ = self.generate_event([self.env])
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
                action_accept_reject, next_state, reward_accept_reject, _ = self.step(t, self.env, PolicyTypes.ACCEPT)

                # Store the gradient of the policy evaluated on the taken action A(t) given the state S(t)
                # Goal: be able to compute e.g. the average policy gradient when learning at the end of the simulation
                gradient_for_action = agent.getLearnerP().getPolicy().getGradient(action_accept_reject, state)
                agent.getLearnerP().record_gradient(state, action_accept_reject, gradient_for_action)

                # Assign the job to a server if accepted
                if action_accept_reject == Actions.ACCEPT:
                    # Assign the job just accepted to a server and update the next_state value
                    # (because the next_state from the ACCEPT action has not been updated above
                    # as we need to know to which server the accepted job is assigned in order
                    # to know the next state of the queue environment)
                    _, next_state, reward_assign, _ = self.step(t, self.env, PolicyTypes.ASSIGN)
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
                self.update_trajectory(agent, t_learn * (self.dict_nsteps['queue'] + 1) + t, t, state, action, reward)
                # Update the average reward (just for information purposes)
                agent.getLearnerV().updateAverageReward(t, reward)

                done = self.check_done(tmax, t, state, action, reward)
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
                # i.e. the reward when a job is serviced is 0).
                reward = np.nan
                if event_prev == Event.BIRTH or event_prev is None:
                    # Add the state PRIOR to the FIRST DEATH after a BIRTH (or the very first DEATH event
                    # if that is the first event that occurs in the simulation), so that we record the state
                    # to which the system went just after the latest BIRTH (so that the plot that is generated
                    # at the end showing the states of the system at each time step does not raise suspicion
                    # because the buffer size doesn't change between two consecutive time steps --which would be
                    # inconsistent with the fact that a new time step is defined when a new job arrives).
                    self.update_trajectory(agent, t_learn * (self.dict_nsteps['queue'] + 1) + t, t, state, action, reward)

            if self.debug:
                print("{} | t={}: event={}, action={} -> state={}, reward={}".format(state, t, event, action, next_state, reward), end="\n")

            event_prev = event

        # DONE
        if verbose and np.mod(t_learn, verbose_period) == 0:
            print("==> agent ENDS at time t={} at state {} coming from state = {}, action = {}, reward = {}, gradient = {})" \
                    .format(t, self.env.getState(), state, action, reward, gradient_for_action))

        return t

    def estimate_stationary_probability_mc(self):
        """
        Monte-Carlo estimation of the stationary probability of K-1 based on the observed trajectory"

        Return: dict
        Dictionary with the estimated stationary probability for buffer size K-1, where K is the first integer
        larger than theta + 1 --being theta the parameter of the acceptance linear step policy.
        """
        # TODO: Estimate the stationary probability in MC using the continuous-time Markov Chain (so that every touch of the state whose probability is of interest is used in the estimation!
        # (not only those discrete time steps where the action is not None as is done now... which may make the stationary probability be estimated as 0.0 even when the continuous-time process touched the state of interest!)
        states = [s for s, a in zip(self.learnerV.getStates(), self.learnerV.getActions()) if a is not None]
        #actions = [a for a in self.learnerV.getActions() if a is not None]
        buffer_sizes = [self.env.getBufferSizeFromState(s) for s in states]
        # print("buffer sizes, actions = {}".format(np.c_[buffer_sizes, actions]))
        K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()
        probas_stationary = dict({K-1: np.sum([1 for bs in buffer_sizes if bs == K - 1]) / len(buffer_sizes)})

        return probas_stationary

    @measure_exec_time
    def run_simulation_fv(self, t_learn, buffer_size_absorption, proba_surv, expected_absorption_time,
                                verbose=False, verbose_period=1):
        """
        Runs the simulation using Monte-Carlo

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        buffer_size_absorption: non-negative int
            Buffer size at which the particle is absorbed.

        proba_surv: pandas data frame
            Probability of survival given the process started at the activation set, Pr(T>t / s in activation set),
            used to estimate the blocking probability for each blocking buffer size.
            Typically this estimation is obtained by running a Monte-Carlo simulation of the queue.
            It should be a pandas data frame with at least two columns:
            - 't': the times at which the survival probability is estimated.
            - 'P(T>t)': the survival probability for each t.

        expected_absorption_time: positive float
            Expected absorption cycle time E(T_A) used to estimate the blocking probability
            for each blocking buffer size.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: int
        The last time step of the simulation process.
        """
        #---------------------------------- Auxiliary functions ------------------------------#
        def get_blocking_buffer_sizes():
            # TODO: (2021/12/08) These buffer sizes should be derived from the buffer sizes where the acceptance policy is > 0
            K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()
            return [K-1, K]

        def update_phi(phi):
            """
            Updates the conditional probability of each buffer size of interest, which are given by the keys
            of the dictionary of the input parameter phi, which is updated.

            Arguments:
            phi: dict
                Dictionary of lists, where the list for each entry (the buffer size of interest)
                contains the empirical distribution at each time when a change happens.
                This input parameter is updated by the function.

            Return: bool
            Whether any of the newly computed Phi values is non-zero. This can be used to decide whether we need to
            compute the integral that gives rise to the Fleming-Viot estimate of the stationary probability, or we
            already know that the integral is already 0 because all the Phi(t) values are 0.
            """
            any_new_value_gt_0 = False
            for bs in phi.keys():
                phi[bs] += [empirical_mean(bs)]
                if phi[bs][-1] > 0.0:
                    any_new_value_gt_0 = True
            return any_new_value_gt_0

        def empirical_mean(buffer_size):
            "Compute the proportion of environments/particles at the given buffer size"
            return np.mean([int(bs == buffer_size) for bs in [env.getBufferSizeFromState(env.getState()) for env in self.envs]])

        def reactivate_particle(idx_particle):
            """
            Chooses a particle to reactivate a particle that has been absorbed

            Reactivation means that the state of the particle is set to be the state of the chosen particle.

            Arguments:
            idx_particle: int
                The index of the particle to be reactivated.

            Return: int
            The index of the particle to which it is reactivated.
            """
            # Select a reactivation particle out of the other N-1 particles
            assert self.N > 1, "There is more than one particle in the system (N={})".format(self.N)
            idx_reactivate = np.random.randint(0, self.N-1)
            if idx_reactivate >= idx_particle:
                # The chosen particle is beyond the particle to reactivate
                # => increase the index of the particle by 1 so that we choose the correct particle
                idx_reactivate += 1
            assert idx_reactivate < self.N, "The reactivation particle ID ({}) is < N (={}) for particle with ID = {}" \
                                            .format(idx_reactivate, self.N, idx_particle)

            self.envs[idx_particle].setState( self.envs[idx_reactivate].getState() )

            return idx_reactivate

        def estimate_stationary_probability(t, phi, proba_surv, expected_absorption_time):
            """
            Estimates the stationary probability for each buffer size of interest in phi using the Fleming-Viot estimator

            Arguments:
            t: list
                Times at which the empirical distribution phi is measured.

            phi: dict of lists
                Empirical distribution of buffer sizes of interest which are the keys of the dictionary.
                The list in each dictionary entry should have the same length as `t`.

            proba_surv: pandas data frame
                Data frame with at least the following two columns:
                - 't': times at which the survival probability is estimated
                - 'P(T>t)': the survival probability estimate for the corresponding 't' value given the process
                started at the stationary activation distribution of states.

            expected_absorption_time: float
                Estimated expected absorption cycle time, i.e. the expected time the queue system takes in a
                re-absorption cycle when starting at the stationary absorption distribution of states.

            Return: tuple of dict
            Duple with two dictionaries indexed by the buffer sizes (bs) of interest with the following content:
            - the stationary distribution
            - the value of the integral P(T>t)*Phi(t,bs)
           """
            # Create a data frame from t and phi where each buffer size in which phi is measured is a different column
            df_phi = pd.DataFrame({'t': t})
            buffer_sizes_of_interest = list( phi.keys() )
            for bs in buffer_sizes_of_interest:
                df_phi[bs] = phi[bs]
            df_phi.columns = pd.MultiIndex.from_arrays([['t'] + ['Phi']*len(buffer_sizes_of_interest), [''] + buffer_sizes_of_interest])

            # Merge the times where Phi(t,bs) and P(T>t) are measured
            df_proba_surv_phi = merge_proba_survival_and_phi(proba_surv, df_phi)

            # Stationary probability for each buffer size of interest
            probas_stationary, integrals = estimate_proba_stationary(df_proba_surv_phi, expected_absorption_time)

            return probas_stationary, integrals

        def merge_proba_survival_and_phi(proba_surv, df_phi):
            """
            Merges the the survival probability and the empirical distribution of buffer sizes of interest
            on a common set of time values into a data frame.

            Arguments:
            proba_surv: pandas data frame
                Data frame containing the time and P(T>t) survival probability given activation.

            df_phi: pandas data frame
                Data frame containing the time and the Phi(t,bs) value for each buffer size of interest bs.

            return: pandas data frame
            Data frame with the following columns:
            - 't': time at which a change in any of the input quantities happens
            - 'P(T>t)': survival probability given the process started at the stationary activation distribution of states.
            - 'Phi(t,bs)': empirical distribution for each buffer size of interest given the process started
            at the stationary activation distribution of states.
            """
            phi_by_t = dict()
            buffer_sizes_of_interest = list( df_phi['Phi'].columns )
            for bs in buffer_sizes_of_interest:
                # Merge the time values at which the survival probability is measured (changes) with the time values
                # at which the empirical distribution Phi is measured for each buffer size of interest.
                t, proba_surv_by_t, phi_by_t[bs] = merge_values_in_time(list(proba_surv['t']), list(proba_surv['P(T>t)']),
                                                                        list(df_phi['t']), list(df_phi['Phi'][bs]), unique=False)

            #-- Merged data frame
            # Initialize with P(T>t)
            # Note: We construct the data frame one column at a time because when defining both columns together
            # in a single call to pd.DataFrame() the order of the columns is not guaranteed!
            df_merged = pd.DataFrame({'t': t})
            df_merged = pd.concat([df_merged, pd.DataFrame({'P(T>t)': proba_surv_by_t})], axis=1)

            # Append Phi for each buffer size of interest
            for bs in buffer_sizes_of_interest:
                df_merged = pd.concat([df_merged, pd.DataFrame({bs: phi_by_t[bs]})], axis=1)

            # Create hierarchical indexing in the columns so that we can access each Phi(t,bs) as df_merged['Phi'][bs][t]
            df_merged.columns = pd.MultiIndex.from_arrays([list(df_merged.columns[:2]) + ['Phi']*len(buffer_sizes_of_interest), ['', ''] + buffer_sizes_of_interest])

            if False:
                print("Survival Probability and Empirical Distribution for each buffer size of interest :")
                print(df_merged)

            return df_merged

        def estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time):
            """
            Computes the stationary probability for each buffer size of interest via Approximation 1 in Matt's draft"

            :param df_phi_proba_surv: pandas data frame
                Data frame with the survival probability P(T>t) and the empirical distribution for each buffer size
                of interest on which the integral that leads to the Fleming-Viot estimation of the stationary
                probability of the buffer size is computed.
            :param expected_absorption_time: float
                Estimated expected absorption cycle time.
            :return: tuple of dict
            Duple with two dictionaries indexed by the buffer sizes (bs) of interest with the following content:
            - the stationary distribution
            - the value of the integral P(T>t)*Phi(t,bs)
            """
            if expected_absorption_time <= 0.0 or np.isnan(expected_absorption_time) or expected_absorption_time is None:
                raise ValueError("The expected absorption time must be a positive float ({})".format(expected_absorption_time))

            probas_stationary = dict()
            integrals = dict()
            for bs in df_phi_proba_surv['Phi'].columns:
                # Integrate => Multiply the survival, the empirical distribution Phi, delta(t) and SUM
                integrals[bs] = 0.0
                for i in range(0, df_phi_proba_surv.shape[0] - 1):
                    integrals[bs] += (df_phi_proba_surv['P(T>t)'].iloc[i] * df_phi_proba_surv['Phi'][bs].iloc[i]) * (df_phi_proba_surv['t'].iloc[i+1] - df_phi_proba_surv['t'].iloc[i])
                if True:
                    print("integrals[{}] = {:.3f}".format(bs, integrals[bs]))
                probas_stationary[bs] = integrals[bs] / expected_absorption_time

            return probas_stationary, integrals

        def estimate_average_reward(phi, probas_stationary):
            """
            Computes the average reward using the stationary probability of each buffer size of interest,
            the policy, and the associated reward defined in the queue environment

            :param phi: dict
                Dictionary with the empirical distribution of the buffer sizes of interest which are
                the dictionary keys. The empirical distribution gives the probability of those buffer sizes
                at each time, conditional to survival up to that time.
            :param probas_stationary: dict
                Dictionary with the estimated stationary probability of the buffer sizes of interest which are
                the dictionary keys.
            :return: float
                Estimated average reward based on a stationary evolution of the system and the reward function
                stored in the queue environment.
            """
            average_reward = 0.0
            for bs in phi.keys():
                average_reward += self.compute_reward_for_buffer_size(bs) * \
                                  self.learnerP.getPolicy().getPolicyForAction(Actions.REJECT, state) * \
                                  probas_stationary[bs]
            return average_reward
        #---------------------------------- Auxiliary functions ------------------------------#

        #---------------------------- Check input parameters ---------------------------------#
        if buffer_size_absorption < 0 \
            or not isinstance(buffer_size_absorption, int) and not isinstance(buffer_size_absorption, np.int32) \
                and not isinstance(buffer_size_absorption, np.int64):
            raise ValueError("The buffer size for absorption must be integer and >= 0 ({})".format(buffer_size_absorption))

        # Check the survival probability values
        if proba_surv is None and not isinstance(proba_surv, pd.core.frame.DataFrame):
            raise ValueError("The survival probability estimate must be given and be a DataFrame")
        if 't' not in proba_surv.columns or 'P(T>t)' not in proba_surv.columns:
            raise ValueError("The data frame with the estimated survival probability must contain at least columns 't' and 'P(T>t)' (columns: {})" \
                             .format(proba_surv.columns))
        if proba_surv['P(T>t)'].iloc[0] != 1.0:
            raise ValueError("The first value of the survival function must be 1.0 ({:.3f})".format(proba_surv['P(T>t)'].iloc[0]))
        if proba_surv['P(T>t)'].iloc[-1] not in [1.0, 0.0]:
            raise ValueError("The survival function at the last measured time is either 1.0 "
                            "(when no particles have been absorbed) or 0.0 "
                            "(when at least one particle has been absorbed) ({})".format(proba_surv['P(T>t)'].iloc[-1]))
        #---------------------------- Check input parameters ---------------------------------#

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(self.envs):
            # Set the start state of the queue environment so that we start at an activation state
            start_state = self.choose_state_for_buffer_size(env, buffer_size_absorption + 1)
            env.setState((start_state, None))
            assert env.getBufferSizeFromState( env.getState() ) == buffer_size_absorption + 1, \
                    "The start state of all environments/particles must be an activation state (start state of env #{}: {})" \
                    .format(i, env.getState())

        # Buffer sizes whose stationary probability is of interest
        buffer_sizes_of_interest = get_blocking_buffer_sizes()

        # Event times (continuous times at which an event happens)
        # The first event time is 0.0
        event_times = [0.0]

        # Phi(t, bs): Empirical probability of the buffer sizes of interest (bs)
        # at each time when an event happens (ANY event, both arriving job or completed service)
        phi = dict()
        for bs in buffer_sizes_of_interest:
            phi[bs] = [empirical_mean(bs)]

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = -1
        maxtime = proba_surv['t'].iloc[-1]
            ## maxtime: it's useless to go with the FV simulation beyond the maximum observed survival time
            ## because the contribution to the integral used in the estimation of the average reward is 0.0 after that.
        is_any_phi_value_gt_0 = False
        while not done:
            # Generate an event over all possible event types and particles
            # Note: this function takes care of NOT generating a service event when a server is empty.
            time, event, job_class_or_server, idx_particle = self.generate_event()
            assert 0 <= idx_particle < self.N, "The chosen particle for change is between 0 and N-1={} ({})".format(self.N-1, idx_particle)
            # Store the absolute event time
            event_times += [ event_times[-1] + time ]

            # Get the current state of the selected particle because that's the one whose state we are going to possibly change
            state = self.envs[idx_particle].getState()
            if event == Event.BIRTH:
                # The event is an incoming job class
                # => Increment the discrete time step, set the arriving job class in the queue environment, and apply the acceptance policy
                # NOTE: Only BIRTH events mark a new discrete time
                # (this guarantees that the stationary distribution of the discrete-time process coincides with that of the continuous-time process)
                t += 1

                # Store the class of the arriving job
                job_class = job_class_or_server
                self.envs[idx_particle].setJobClass(job_class)

                # Perform a step on the environment by applying the acceptance policy
                action_accept_reject, next_state, reward_accept_reject, _ = self.step(t, self.envs[idx_particle], PolicyTypes.ACCEPT)

                # Assign the job to a server if accepted
                if action_accept_reject == Actions.ACCEPT:
                    # Assign the job just accepted to a server and update the next_state value
                    # (because the next_state from the ACCEPT action has not been updated above
                    # as we need to know to which server the accepted job is assigned in order
                    # to know the next state of the queue environment)
                    _, next_state, reward_assign, _ = self.step(t, self.envs[idx_particle], PolicyTypes.ASSIGN)
                else:
                    reward_assign = 0.0

                action = action_accept_reject
                reward = reward_accept_reject + reward_assign
                if action == Actions.ACCEPT and reward != 0.0:
                    print("--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward))
                    raise ValueError("A non-zero reward is not possible when the action is ACCEPT.")

                # Stop when we reached the maximum observed time when estimating P(T>t)
                # (since going beyond does not contribute to the integral used in the FV estimator)
                done = event_times[-1] > maxtime
            elif event == Event.DEATH:
                # The event is a completed service
                # => Update the state of the queue but do NOT update the discrete-time step
                server = job_class_or_server
                queue_state = self.envs[idx_particle].getQueueState()
                assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"
                next_queue_state = copy.deepcopy(queue_state)
                next_queue_state[server] -= 1
                self.envs[idx_particle].setState((next_queue_state, None))
                next_state = self.envs[idx_particle].getState()

                assert self.envs[idx_particle].getBufferSizeFromState(next_state) == self.envs[idx_particle].getBufferSizeFromState(state) - 1, \
                    "The buffer size after a DEATH decreased by 1: S(t) = {}, S(t+1) = {}" \
                        .format(self.envs[idx_particle].getBufferSizeFromState(state), self.envs[idx_particle].getBufferSizeFromState(next_state))

                # Check if the particle has been ABSORBED
                if self.envs[idx_particle].getBufferSizeFromState(next_state) == buffer_size_absorption:
                    # The particle has been absorbed
                    # => Reactivate it to any of the other particles
                    reactivate_particle(idx_particle)
                    next_state = self.envs[idx_particle].getState()

                # We set the action to None because there is no action by the agent when a service is completed
                action = None
                # We set the reward to NaN because a reward may happen ONLY when a new job arrives.
                # If there are rewards due to serviced jobs, we would impact those rewards into the reward
                # received when a new job arrives (which is not the case at this moment (28-Nov-2021),
                # i.e. the reward when a job is serviced is 0).
                reward = np.nan

            if self.debug:
                print("{} | t={}: event={}, action={} -> state={}, reward={}".format(state, t, event, action, next_state, reward), end="\n")

            is_any_phi_value_gt_0 = max([is_any_phi_value_gt_0, update_phi(phi)])

        # DONE
        if verbose and np.mod(t_learn, verbose_period) == 0:
            print("==> agent ENDS at discrete time t={} (continuous time = {:.1f}, compared to maximum observed time for P(T>t) = {:.1f}) at state {} coming from state = {}, action = {}, reward = {})" \
                    .format(t, event_times[-1], proba_surv['t'].iloc[-1], self.envs[idx_particle].getState(), state, action, reward))
        # Assertion
        for bs in phi.keys():
            assert len(event_times) == len(phi[bs]), "The length of the event times where phi is measured ({}) coincides with the length of phi ({})" \
                                                .format(len(event_times), len(phi))

        if is_any_phi_value_gt_0:
            # Compute the stationary probability of each buffer size bs in Phi(t,bs) using Phi(t,bs), P(T>t) and E(T_A)
            probas_stationary, integrals = estimate_stationary_probability(event_times, phi, proba_surv, expected_absorption_time)

            # Compute the average reward based on the stationary probability and the empirical distribution of each buffer size bs
            average_reward = estimate_average_reward(phi, probas_stationary)
        else:
            probas_stationary = dict()
            integrals = dict()
            for bs in phi.keys():
                probas_stationary[bs] = integrals[bs] = 0.0
            average_reward = 0.0    # The average reward is non-zero only when the stationary probability of at least one buffer size of interest is non-zero

        return t, event_times, phi, probas_stationary, average_reward

    @measure_exec_time
    def estimate_Q_values_until_stationarity(self, t_learn, t_sim_max=50, N=100, verbose=False, verbose_period=1):
        """
        Estimates the state-action values for the case of a parameterized policy defined as a linear function of theta
        with only one buffer size with non-zero derivative.
        The state-action values are computed for the two possible accept/reject actions of an incoming job when
        the queue system is at that buffer size.

        Arguments:
        t_learn: int
            Learning time step.

        t_sim_max: (opt) int
            Maximum discrete time for the simulation, after which stationary is assumed to have been reached.
            default: 50

        N: (opt) int
            Number of samples used to estimate the state-action values.
            default: 100

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - K: the buffer size at which an incoming job is always rejected by the policy (deterministic policy)
        - The state-action value for (K-1, a=Reject)
        - The state-action value for (K-1, a=Accept)
        - N: the number of realizations used to compute the state-action values
        """
        if not isinstance(self.learnerP.getPolicy(), PolQueueTwoActionsLinearStep):
            raise ValueError("The agent's policy must be of type {} ({})" \
                             .format(type(PolQueueTwoActionsLinearStep), type(self.learner.getPolicy())))

        K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()

        # 1) Starting at (K-1, a=1) is like starting at s=K (because the reward of the first action is 0)
        if verbose and np.mod(t_learn, verbose_period) == 0:
            print("\n1) Running simulation on N={} queues for {} time steps to estimate Q(K-1, a=1)...".format(N, t_sim_max))
        start_state = self.choose_state_for_buffer_size(self.env, K)
        Q1 = 0.0
        for _ in range(N):
            agent_Q = copy.deepcopy(self.agent)
            agent_Q.getLearnerV().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            agent_Q.getLearnerP().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            self.run_simulation_mc(t_learn, start_state, t_sim_max=t_sim_max, agent=agent_Q, verbose=False, verbose_period=verbose_period)
            Q1 += np.nansum(agent_Q.getLearnerV().getRewards())  # Note: the rewards are not NaN only when an action was taken (i.e. at job arrivals)
            del agent_Q
        Q1 /= N

        # 2) Starting at (K-1, a=0) is like starting at s=K-1 and adding one reward for the first rejection
        if verbose and np.mod(t_learn, verbose_period) == 0:
            print("2) Running simulation on N={} queues for {} time steps to estimate Q(K-1, a=0)...".format(N, t_sim_max))
        start_state = self.choose_state_for_buffer_size(self.env, K - 1)
        # Reward received from the initial rejection of the job at buffer size K-1
        first_reward = self.compute_reward_for_buffer_size(K - 1)
        Q0 = 0.0
        for _ in range(N):
            agent_Q = copy.deepcopy(self.agent)
            agent_Q.getLearnerV().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            agent_Q.getLearnerP().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            self.run_simulation_mc(t_learn, start_state, t_sim_max=t_sim_max, agent=agent_Q, verbose=False, verbose_period=verbose_period)
            Q0 += np.nansum(agent_Q.getLearnerV().getRewards())  # Note: the rewards are not NaN only when an action was taken (i.e. at job arrivals)
            del agent_Q
        Q0 = Q0 / N + first_reward  # The first reward is the same for all queues because it only depends on the buffer size which is the same for all queues and equal to K-1

        return K, Q0, Q1, N, t_sim_max

    @measure_exec_time
    def estimate_Q_values_until_mixing(self, t_learn, t_sim_max=250, N=100, verbose=False, verbose_period=1):
        """
        Estimates the state-action values for the case of a parameterized policy defined as a linear function of theta
        with only one buffer size with non-zero derivative.
        The state-action values are computed for the two possible accept/reject actions of an incoming job when
        the queue system is at that buffer size.

        Arguments:
        t_learn: int
            Learning time step.

        t_sim_max: (opt) int
            Maximum discrete time for the simulation. If this is reached without mixing, the estimated state-action
            values are set to NaN.
            default: 250

        N: (opt) int
            Number of samples used to estimate the state-action values.
            default: 100

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - K: the buffer size at which an incoming job is always rejected by the policy (deterministic policy)
        - The state-action value for (K-1, a=Reject)
        - The state-action value for (K-1, a=Accept)
        - n: the number of realizations, where trajectories mix, that are used to compute the state-action values
        """
        if not isinstance(self.learnerP.getPolicy(), PolQueueTwoActionsLinearStep):
            raise ValueError("The agent's policy must be of type {} ({})" \
                             .format(type(PolQueueTwoActionsLinearStep), type(self.learner.getPolicy())))

        K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()

        Q0 = 0.0
        Q1 = 0.0
        n = 0           # Number of times Q0 and Q1 can be estimated from mixing trajectories (out of N)
        max_t_mix = 0.0
        for _ in range(N):
            _, Q0_, Q1_, t_mix = self.run_simulation_2_until_mixing(t_learn, t_sim_max=t_sim_max, verbose=False, verbose_period=verbose_period)
            if not np.isnan(Q0_) and not np.isnan(Q1_):
                Q0 += Q0_
                Q1 += Q1_
                n += 1
                max_t_mix = np.max([max_t_mix, t_mix])
        if n == 0:
            warnings.warn("Q_diff cannot be estimated because no mixing occurred for any of the N={} simulated trajectory pairs. It will be estimated as 0.0".format(N))
        else:
            Q0 /= n
            Q1 /= n

        return K, Q0, Q1, n, max_t_mix

    def run_simulation_2_until_mixing(self, t_learn, t_sim_max=None, verbose=False, verbose_period=1):
        """
        Runs the simulation using Monte-Carlo

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        agent: Agent
            Agent object that is responsible of performing the actions on the environment and learning from them.

        t_sim_max: int
            Maximum discrete simulation time allowed for the simulation. This is defined ONLY by job arrivals,
            NOT by service events.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - t: the last time step of the simulation process.
        - Q0: the sum of the rewards observed until mixing starting at state-action (K-1, a=Reject)
        - Q1: the sum of the rewards observed until mixing starting at state-action (K-1, a=Accept)
        - mixing time: discrete time at which the two trajectories mix. This is np.nan if they don't mix.
        """

        # ------------------------------ Auxiliary functions ----------------------------------#
        def apply_event(env, t, event, job_class_or_server):
            """
            Apply the given event, either a BIRTH with the incoming job class or a DEATH event at the given server,
            on the given environment that is currently at discrete time t.

            Arguments:
            env: Queue environment
                Queue environment where the event takes place.

            t: int
                Discrete time at which the environment is when the event occurs.

            event: Event
                Type of event that takes place in the environment, either Event.BIRTH or Event.DEATH.

            job_class_or_server: int
                Index associated to the incoming job class (when event = Event.BIRTH) or to the server that finishes
                serving a job (when event = Event.DEATH).

            Return: tuple
            Tuple with the following elements:
            - t': the new time after the event is applied. This increases only for BIRTH events, i.e. for arrival
            of new jobs.
            - state: S(t), the state at which the environment is prior to applying the action.
            - action: A(t), the action taken by the agent at state S(t).
            - next_state: S(t'), the next state at which the environment is in after taking action A(t).
            - reward: the reward received by the agent for taking action A(t).
            """
            # Get the current state of the selected particle because that's the one whose state we are going to possibly change
            state = env.getState()
            if event == Event.BIRTH:
                # The event is an incoming job class
                # => Increase the discrete time step and set the arriving job class in the queue environment and apply the acceptance policy
                t += 1

                # Store the class of the arriving job
                job_class = job_class_or_server
                env.setJobClass(job_class)

                # Perform a step on the environment by applying the acceptance policy
                action_accept_reject, next_state, reward_accept_reject, _ = self.step(t, env, PolicyTypes.ACCEPT)

                # Assign the job to a server if accepted
                if action_accept_reject == Actions.ACCEPT:
                    # Assign the job just accepted to a server and update the next_state value
                    # (because the next_state from the ACCEPT action has not been updated above
                    # as we need to know to which server the accepted job is assigned in order
                    # to know the next state of the queue environment)
                    _, next_state, reward_assign, _ = self.step(t, env, PolicyTypes.ASSIGN)
                else:
                    reward_assign = 0.0

                action = action_accept_reject
                reward = reward_accept_reject + reward_assign
                if action == Actions.ACCEPT and reward != 0.0:
                    print("--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward))
                    raise ValueError("A non-zero reward is not possible when the action is ACCEPT.")
            else:
                assert event == Event.DEATH
                # The event is a completed service
                # => Update the state of the queue but do not increase t
                server = job_class_or_server
                queue_state = env.getQueueState()
                assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"
                next_queue_state = copy.deepcopy(queue_state)
                next_queue_state[server] -= 1
                env.setState((next_queue_state, None))
                next_state = env.getState()

                assert env.getBufferSizeFromState(next_state) == env.getBufferSizeFromState(
                    state) - 1, \
                    "The buffer size after a DEATH decreased by 1: S(t) = {}, S(t+1) = {}" \
                        .format(env.getBufferSizeFromState(state),
                                env.getBufferSizeFromState(next_state))

                # We set the action to None because there is no action by the agent when a service is completed
                action = None
                # No reward when no action taken
                reward = np.nan

            return t, state, action, next_state, reward

        mixing = lambda s0, a0, s1, a1: s0 == s1 and a0 is not None and a1 is not None and a0 == a1
        # ------------------------------ Auxiliary functions ----------------------------------#

        # Create the two queue environments on which the simulation takes place
        envs = [copy.deepcopy(self.env), copy.deepcopy(self.env)]

        # Set the state of each environment AFTER taking the respective action at time t=0.
        # This is K-1 when the first action is Reject and K when the first action is Accept.
        K = self.learnerP.getPolicy().getBufferSizeForDeterministicBlocking()

        # History of states, actions and rewards of each environment (2 x <undef> list)
        # Used for informational purposes, but actually we don't need to store the history because we are computing
        # Q0 and Q1 on the fly.
        times = [[0],
                 [0]]
        states = [[(K-1, None)],
                  [(K-1, None)]]
        actions = [[Actions.REJECT],
                   [Actions.ACCEPT]]
        rewards = [[self.compute_reward_for_buffer_size(K-1)],
                   [0.0]]

        # Set the NEW state of the environments once the above actions have been taken on the respective states
        envs[0].setState((K-1, None))   # Environment that starts at REJECTION
        envs[1].setState((K, None))     # Environment that starts at ACCEPTANCE

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = 0                   # We start at 0 because we have already stored the state for t=0 above
        t0 = 0; t1 = 0;         # We keep track of the time step of each queue which only changes
                                # when there is an incoming job, and thus we can compare their state at the same
                                # discrete time value.
        maxtime = t_sim_max     # Max discrete time, in case there is no mixing of trajectories
        # Initialize the state-action values with the first observed reward (at t=0)
        Q0 = rewards[0][0]
        Q1 = rewards[1][0]
        while not done:
            # Generate events for each queue environment until the discrete time changes,
            # so that we can compare their state-action at the same discrete time value and find the mixing time.
            # Note: the generate_events() function takes care of NOT generating a service event when a server is empty.
            while t0 == t:
                time0, event0, job_class_or_server0, _ = self.generate_event([envs[0]])
                t0, state0, action0, next_state0, reward0 = apply_event(envs[0], t, event0, job_class_or_server0)
            while t1 == t:
                time1, event1, job_class_or_server1, _ = self.generate_event([envs[1]])
                t1, state1, action1, next_state1, reward1 = apply_event(envs[1], t, event1, job_class_or_server1)
            assert t0 == t1 and t0 == t + 1, \
                "The discrete time step is the same in both queues and has increased by one w.r.t. the global time step t (t={:}, t0={:}, t1={:})" \
                .format(t, t0, t1)
            t = t0

            # Update the history
            times[0] += [time0]; times[1] += [time1]
            states[0] += [state0]; states[1] += [state1]
            actions[0] += [action0]; actions[1] += [action1]
            rewards[0] += [reward0]; rewards[1] += [reward1]

            # Update the state-action values
            assert not np.isnan(reward0) and not np.isnan(reward1), "The reward given by each queue are not NaN (r0={}, r1={})".format(reward0, reward1)
            Q0 += reward0
            Q1 += reward1

            # Check if the trajectories meet at the same state-action
            if t > maxtime or mixing(state0, action0, state1, action1):
                # Either we reached the maximum simulation time allowed or mixing occurs
                # => Stop the simulation
                done = True

            if self.debug:
                print("t={}: states = [{}, {}] events={} actions={} -> states={} rewards={}" \
                      .format(t, states[-2][0], states[-2][1], [event0, event1], [action0, action1], [state0, state1], [reward0, reward1]), end="\n")

        # DONE
        if verbose and np.mod(t_learn, verbose_period) == 0:
            print("==> simulation ENDS at discrete time t={} at states = {} coming from states = {}, actions = {} rewards = {})" \
                    .format(t, [state0, state1], [states[-2][0], states[-2][1]], [action0, action1], [reward0, reward1]))

#        plt.figure()
#        print("Mixing time t={}, Q0={}, Q1={}".format(t, Q0, Q1))
#        print(np.c_[[envs[0].getBufferSizeFromState(s) for s in states[0]], actions[0], rewards[0],
#                    [envs[1].getBufferSizeFromState(s) for s in states[0]], actions[1], rewards[1]])
#        plt.plot([envs[0].getBufferSizeFromState(s) for s in states[0]], 'b.-')
#        plt.plot([envs[1].getBufferSizeFromState(s) for s in states[1]], 'r.-')
#        plt.show()
#        input("Press ENTER...")

        if not mixing(state0, action0, state1, action1):
            # The trajectories didn't mix
            # => Discard the estimations of Q0 and Q1 because they would be misleading as the trajectories didn't mix
            Q0 = Q1 = np.nan
            t_mix = np.nan
        else:
            t_mix = t

        return t, Q0, Q1, t_mix

    def choose_state_for_buffer_size(self, env, buffer_size):
        """
        Chooses randomly a server in the queue environment to have its queue size equal to the given buffer size,
        while the queue size of all other servers are set to 0, so that the buffer size of the system is equal
        to the given buffer size.

        Arguments:
        env: Queue environment
            Queue environment containing the servers.

        buffer_size: int
            Buffer size to be set in the system.

        Return: list
        List containing the state to set the queue environment at that guarantees that the system's buffer size is
        equal to the given buffer size.
        """
        state = [0] * env.getNumServers()
        state[np.random.randint(0, len(state))] = buffer_size
        return state

    def generate_event(self, envs=None):
        """
        Generates the next event for the queue system **by taking into account the server's queue sizes**,
        meaning that no completed service can occur on an empty server.

        The next event can be either:
        - an incoming job (of a given class)
        - a completed service in a server in the queue system having at least one job

        envs: (opt) list of queue environments
            List of queue environments to consider for the generation of the event, i.e. where the event could be
            generated.
            default: None, which means that self.envs is used

        Return: tuple
        Tuple containing the following elements:
        - the time of the event
        - the event type (Event.BIRTH or Event.DEATH)
        - an index that represents either:
            - the job class, if the generated event is an arriving job
            - the server, if the generated event is a completed service
        - the environment/particle where the event takes place
        """
        #-- Parse input parameters
        if envs is None:
            envs = self.envs

        # Define a 2D numpy array where the different environments/particles are in the first dimension
        # and the job classes and servers are in the second dimension. The value of each entry is
        # either the job class arrival rate or the server service rate, depending on the column position
        # (first come the job classes and then the servers).
        # A service rate is NaN when the server size is 0, so that the server cannot be selected to experience
        # a service rate.
        valid_rates = np.array(
                        [env.getJobClassRates() + [r if s > 0 else np.nan for r, s in zip(env.getServiceRates(), env.getQueueState())]
                         for env in envs])

        # Generate the event time and the index on the rates array to which it is associated
        event_time, event_idx = generate_min_exponential_time(valid_rates.flatten())

        # Convert the linear index back to the 2D index in valid_rates where the indices represent:
        # (selected env/particle, selected rate)
        idx_env, job_class_or_server = index_linear2multi(event_idx, valid_rates.shape)

        # Define whether the generated event is an incoming job class or a completed service
        # and return a different tuple accordingly.
        # It is assumed that the number of job classes and servers are the same in each environment.
        n_job_classes = self.env.getNumJobClasses()
        if job_class_or_server < n_job_classes:
            # The event is an incoming job class
            job_class = job_class_or_server
            return event_time, Event.BIRTH, job_class, idx_env
        else:
            # The event is a completed service event
            idx_server = job_class_or_server - n_job_classes
            return event_time, Event.DEATH, idx_server, idx_env

    def step(self, t, env, policy_type):
        """
        Arguments:
        t: int
            Current queue transition associated to the current step.

        env: environment
            The environment where the agent acts.

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
        action, observation, reward, info = self.agent.act(env, policy_type)

        return action, observation, reward, info

    def update_trajectory(self, agent, t_total, t_sim, state, action, reward):
        """

        Arguments:
        agent: Agent
            Agent that is responsible of performing the actions on the environment and learning from them
            on which the trajectory to update is stored.

        t_total: int
            Total time, used to store the trajectory for the policy learning.
            Normally this would be the total simulation time computed from the learning steps.

        t_sim: int
            Simulation time, used to store the trajectory for the value function learning.

        state: State (however the state is defined, e.g. as the buffer size)
            State at the given time to store in the trajectory.

        action: Action
            Action taken at the given time to store in the trajectory.

        reward: float
            Reward received by the agent after taking the given action at the given state, to store in the trajectory.
        """
        agent.getLearnerV().update_trajectory(t_sim, state, action, reward)
        # DM-2021/11/28: This assertion is no longer true because we are now storing in the trajectory ALSO the states
        # occurring just before the FIRST DEATH event happening after a BIRTH event (so that we can show it in the
        # trajectory plot we show at the end of the simulation and thus avoid the suspicion that something is wrong
        # when we observe no change in the buffer size from one time step to the next --given that a new time step
        # is created ONLY when a new job arrives (i.e. a BIRTH event occurs)
        #assert action is not None, "The action is not None when learning the policy"
        agent.getLearnerP().update_trajectory(t_total, state, action, reward)

    def compute_reward_for_buffer_size(self, bs):
        "Computes the reward received when blocking at the given buffer size"
        # *** We assume that the state of the environment is a duple: (list with server queue sizes, job class of latest arriving job) ***
        state = ([0] * self.env.getNumServers(), None)
        # We set the first server to have occupancy equal to the buffer size of interest
        # as we assume that the reward function ONLY depends on the buffer size, and NOT on the actual state
        # of the multi-server system.
        state[0][0] = bs
        reward_func = self.env.getRewardFunction()
        reward = reward_func(self.env, state, Actions.REJECT, state)

        return reward

    @measure_exec_time
    def learn(self, agent, t, proba_stationary=None, Q_values=None):
        """
        Learns the value functions and the policy at time t

        Arguments:
        agent: Agent
            Agent that is responsible of performing the actions on the environment and learning from.

        t: int
            Time (iteration) in the queue simulation time scale at which learning takes place.
            In Monte-Carlo learning, this time would be the end of the queue simulation.

        proba_stationary: (opt) float between 0 and 1 (used when estimating the theoretical grad(V))
            Estimated stationary probability of buffer size = K-1.
            default: None

        Q_values: (opt) list or numpy array (used when estimating the theoretical grad(V))
            Estimated state-action values for (K-1, a=1) and (K-1, a=0)
            default: None
        """
        # TODO: (2021/12/13) Refactor the signature of the method so that we pass a parameter that states whether we want to use the estimation of the theoretical grad(V) or go for the REINFORCE algorithm

        # TODO: (2021/11/03) Think if we can separate the learning of the value function and return G(t) from the learning of the policy (theta)
        # Goal: avoid computing G(t) twice, one at learnerV.learn() and one at learnerP.learn()

        # Store the alpha used for the current learning step
        agent.getLearnerP().store_learning_rate()

        if proba_stationary is None or Q_values is None:
            agent.getLearnerV().learn(t)  # UNCOMMENTED ON SAT, 27-NOV-2021
            theta_prev, theta, V, gradV, G = agent.getLearnerP().learn(t)
            #theta_prev, theta, V, gradV, G = agent.getLearnerP().learn_TR(t)
        else:
            # When learning via the theoretical grad(V) expression we don't need to estimate the average reward!
            # (because grad(V) is the difference between two state-action values, and the average reward affects both the same way)
            # This is the option to use when learning via Fleming-Viot because in that case the return G(t)
            # --used when learning with learn() or learn_TR() or learn_linear_theoretical()-- is NOT meaningful
            # because, either it is empty as the rewards are not recorded while running the Fleming-Viot process,
            # or it does not reflect the actual values of each state-value of the original system.
            theta_prev, theta, V, gradV, G = agent.getLearnerP().learn_linear_theoretical_from_estimated_values(t, proba_stationary, Q_values)
        self.alphas += [agent.getLearnerP().getLearningRate()]
        self.thetas += [theta_prev]
        self.thetas_updated += [theta]
        self.proba_stationary += [proba_stationary]
        self.Q_diff += [Q_values[1] - Q_values[0]]
        self.V += [V]               # NOTE: This is NOT always the average value, rho... for instance, NOT when linear_theoretical_from_estimated_values() is called above
        self.gradV += [gradV]
        self.G = G

        agent.getLearnerP().update_learning_time()
        print("*** Learning-time updated: time = {} ***".format(agent.getLearnerP().getLearningTime()))
        agent.getLearnerP().update_learning_rate_by_episode()

    def check_done(self, tmax, t, state, action, reward):
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

    # ------ GETTERS ------#
    def getJobRatesByServer(self):
        return self.job_rates_by_server


if __name__ == "__main__":
    import runpy

    runpy.run_path("../../setup.py")

    test = False

    # --------------- Unit tests on methods defined in this file ------------------ #
    if test:
        from environments.queues import EnvQueueSingleBufferWithJobClasses

        # ---------------- generate_event() in SimulatorQueue --------------------- #
        print("Testing generate_event() in SimulatorQueue class:")

        # Setup the simulator with the minimum required information
        capacity = 10
        nservers = 4
        job_class_rates = [0.7, 0.5]
        service_rates = [1.0, 0.8, 1.3, 0.75]
        job_rates_by_server = compute_job_rates_by_server(job_class_rates, nservers, [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.4, 0.1]])
        queue = QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, None, None)
        simul = SimulatorQueue(env_queue_mm, None, None, N=1)

        np.random.seed(1717)
        time, event, job_class_or_server, idx_env = simul.generate_event()
        print("Observed minimum time {} for an {} associated to job class (if BIRTH) or server (if DEATH) = {}".format(time, event, job_class_or_server))
        time_expected, event_expected, job_class_or_server_expected, idx_env_expected = 0.36603, Event.BIRTH, 0, 0
        assert np.allclose(time, time_expected, atol=1E-5)
        assert event == event_expected
        assert job_class_or_server == job_class_or_server_expected
        assert idx_env == idx_env_expected

        # Repeat the realization to check that we always get a BIRTH event because by default the queue sizes of the servers are initialized at 0
        N = 10
        for _ in range(N):
            _, event, _, _ = simul.generate_event()
            assert event == Event.BIRTH

        #-- Repeat the realization to check the distribution of event types and selected job classes or servers
        # We set a random seed so that we embrace many more test cases as every time we run this test we get new realizations!
        np.random.seed(None)
        # Set two of the servers to a positive queue size
        simul.envs[0].setState(([0, 1, 0, 2], None))

        # Keep track of the number of events observed for each rate (either a job class rate or a service rate)
        rates = job_class_rates + service_rates
        valid_rates = job_class_rates + [r if s > 0 else np.nan for r, s in zip(simul.envs[0].getServiceRates(), simul.envs[0].getQueueState())]
        nevents_by_rate = np.zeros(len(rates))
        N = 100
        # Expected proportion of events for each exponential
        p = np.array( [r / np.nansum(valid_rates) if r > 0 else 0.0 for r in valid_rates] )
        for _ in range(N):
            time, event, job_class_or_server, _ = simul.generate_event()
            if event == Event.BIRTH:
                # The BIRTH events are associated to indices at the beginning of the rates list
                nevents_by_rate[job_class_or_server] += 1
            elif event == Event.DEATH:
                # The DEATH event are associated to indices at the end of the rates list
                # so we sum the returned index to the number of job class rates.
                nevents_by_rate[len(job_class_rates) + job_class_or_server] += 1
        # Observed proportions of events by server and type of event
        phat = 1.0 * nevents_by_rate / N
        se_phat = np.sqrt(p * (1 - p) / N)
        print("EXPECTED / OBSERVED / SE proportions of events by rate on N={}:\n{}".format(N, np.c_[p, phat, se_phat]))
        assert np.allclose(phat, p,
                           atol=3 * se_phat)  # true probabilities should be contained in +/- 3 SE(phat) from phat
        # ---------------- generate_event() in SimulatorQueue --------------------- #

    # --------------- Unit tests on methods defined in this file ------------------#
    if not test and False:
        #--- Test the Simulator class ---#
        # NOTE: (2021/10/19) This test works
        # from Python.lib import environments, agents
        from environments import gridworlds
        from agents import GenericAgent
        from agents.policies import random_walks
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

    if not test and True:
        start_time_all = timer()

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
        theta_start = 39.3  # 1.3, 11.3  # IMPORTANT: This value should NOT be integer, o.w. the policy gradient will always be 0 regardless of the state at which blocking occurs
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue_mm, theta_start), PolicyTypes.ASSIGN: None})
        #policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLogit(env_queue_mm, theta_start, beta=1.0), PolicyTypes.ASSIGN: None})

        # Learning method (MC or FV)
        n_particles_fv = 400; t_sim = 500
        #n_particles_fv = 800; t_sim = 800
        #n_particles_fv = 800; t_sim = 1000
        learning_method = LearningMethod.MC; nparticles = 1; plot_trajectories = False; symbol = 'b.-'
        #learning_method = LearningMethod.FV; nparticles = n_particles_fv; plot_trajectories = False; symbol = 'g.-'

        # Simulator on a given number of iterations for the queue simulation and a given number of iterations to learn the policy
        t_learn = 200 #50
        nmeantimes = t_sim * job_class_rates[0] # Simulation time to use in the FV estimation when estimating P(T>t) and E(T_A)

        # Learners definition
        fixed_window = False
        alpha_start = 100.0 #/ t_sim  # Use `/ t_sim` when using update of theta at each simulation step (i.e. LeaPolicyGradient.learn_TR() is called instead of LeaPolicyGradient.learn())
        adjust_alpha = True
        min_time_to_update_alpha = t_learn / 2.5
        alpha_min = 0.1
        gamma = 1.0
        if learning_method == LearningMethod.FV:
            learnerV = LeaFV(env_queue_mm, gamma=gamma)
        else:
            # Extend the simulation time in the MC case to as many particles used in the FV case
            t_sim *= n_particles_fv
            learnerV = LeaMC(env_queue_mm, gamma=gamma)

        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(env_queue_mm, policies[PolicyTypes.ACCEPT], learnerV,
                                                           alpha=alpha_start, adjust_alpha=adjust_alpha,
                                                           min_time_to_update_alpha=min_time_to_update_alpha,
                                                           alpha_min=alpha_min, fixed_window=fixed_window)})
        agent_gradient = AgeQueue(env_queue_mm, policies, learners)

        seed = 1717
        dict_nsteps = dict({'queue': t_sim, 'learn': t_learn})
        dict_params_simul = dict({
            'nparticles': nparticles,
            'nmeantimes': nmeantimes,
            'buffer_size_activation': np.nan,   # This value will be defined by the simulation process as it is updated at each updated theta value
            'seed': np.nan
            })
        dict_params_info = dict({'plot': False, 'log': False})
        simul = SimulatorQueue(env_queue_mm, agent_gradient, dict_nsteps, N=nparticles, debug=False)
        learnerV, learnerP, df_learning = simul.run(dict_params_simul, dict_params_info=dict_params_info, seed=seed, verbose=True)

        # Save the estimation of G(t) for the last learning step to a file
        #file_results_G = "G.csv"
        #pd.DataFrame({'G': simul.G}).to_csv(file_results_G)

        #-- Plot theta and the gradient of the value function
        SET_YLIM = False

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
        ax.set_xlim((0, ax.get_xlim()[1]))
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
        if plot_trajectories:
            assert N == 1, "The simulated system has only one particle (N={})".format(N)
            # NOTE: (2021/11/27) I verified that the values of learnerP.getRewards() for the last learning step
            # are the same to those for learnerV.getRewards() (which only stores the values for the LAST learning step)
            plt.figure()
            # Times at which the trajectory of states is recorded
            times = learnerP.getTimes()
            times_unique = np.unique(times)
            assert len(times_unique) == len(learnerP.getPolicy().getThetas()) - 1, \
                "The number of unique learning times ({}) is equal to the number of theta updates ({})".format(len(times_unique), len(learnerP.getPolicy().getThetas()) - 1)
                ## Note that we subtract 1 to the number of learning thetas because the first theta stored in the policy object is the initial theta before any update
            plt.plot(np.r_[0.0, times_unique], learnerP.getPolicy().getThetas(), 'b.-')
                ## We add 0.0 as the first time to plot because the first theta value stored in the policy is the initial theta with which the simulation started
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
            ax.plot(times_sim_starts, [buffer_sizes[t] for t in times_sim_starts], 'gx')
            ax.set_xlabel("time step")
            ax.set_ylabel("theta")

            # Secondary plot showing the rewards received
            ax2 = ax.twinx()
            ax2.plot(times, -np.array(learnerP.getRewards()), 'r-', alpha=0.3)
            # Highlight with points the non-zero rewards
            ax2.plot(times, [-r if r != 0.0 else None for r in learnerP.getRewards()], 'r.')
            ax2.set_ylabel("Reward")
            ax2.set_yscale('log')
            plt.title("Optimum Theta = {}, Theta start = {}, t_sim = {:.0f}, t_learn = {:.0f}, fixed_window={}".format(COST_EXP_BUFFER_SIZE_REF, theta_start, t_sim, t_learn, fixed_window))
        else:
            plt.figure()
            plt.plot(learnerP.getPolicy().getThetas(), symbol)
            plt.title("Method: {}, Optimum Theta = {}, Theta start = {}, t_sim = {:.0f}, t_learn = {:.0f}, fixed_window={}" \
                      .format(learning_method.name, COST_EXP_BUFFER_SIZE_REF, theta_start, t_sim, t_learn, fixed_window))
            ax = plt.gca()
            ylim = ax.get_ylim()
            ax.set_ylim((0, ylim[1]))

        end_time_all = timer()
        elapsed_time_all = end_time_all - start_time_all
        print("\n+++ OVERALL execution time: {:.1f} min, {:.1f} hours".format(elapsed_time_all/60, elapsed_time_all/3600))
