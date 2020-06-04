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
and getters:
    - getNumStates()
    - getNumActions()
    - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
        (responsible for defining the initial state when running the simulation)
    - getState() --> returns the current state of the environment
and setters:
    - setInitialStateDistribution()

The environments of class gym.toy_text.Env satisfy the above conditions.
"""


import numpy as np
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())

if __name__ == "__main__":
    from environments import EnvironmentDiscrete
    from agents import GenericAgent
    from agents.learners.td import LeaTDLambdaAdaptive
    from utils.computing import rmse
else:
    # These relative imports are only accepted when we compile the file as a module
    from .environments import EnvironmentDiscrete
    from .agents import GenericAgent
    from .agents.learners.td import LeaTDLambdaAdaptive
    from .utils.computing import rmse

# TODO: (2020/05) Rename this class to SimulatorDiscrete as it simulates on a discrete environment
class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a given environment `env`
    and an `agent`.
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
            Index corresponding to the starting state.

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
                - state value function estimate for each state at the end of the last episode (`nepisodes`).
                - number of visits to each state at the end of the lastt episode.
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
            # Plot the true state value function (to have it as a reference already
            plt.plot(self.env.all_states, self.env.getV(), '.-', color="blue")
            
            if state_observe is not None:
                fig_RMSE_state = plt.figure()

        # Define initial state
        nS = self.env.getNumStates()
        if start is not None:
            if not (isinstance(start, int) and 0 <= start and start < nS):
                raise Warning("The `start` parameter ({}) must be an integer number between 0 and {}.\n" \
                              "A start state will be selected based on the initial state distribution of the environment." \
                              .format(start, nS-1))
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
                state_observe = int(nS/2)
                raise Warning("The `state_observe` parameter must be an integer number between 0 and {}.\n" \
                              "The state whose index falls in the middle of the state space will be observed.".format(nS-1))

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
                print("Episode {} of {} running...".format(episode+1, nepisodes))
            if self.debug:
                print("\nStarts at state {}".format(self.env.getState()))
                print("\tState value function at start of episode:\n\t{}".format(learner.getV().getValues()))

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
                    print("--> Done [{} iterations] at state {} with reward {}".
                          format(t+1, self.env.getState(), reward))
                    print("\tUpdating the value function at the end of the episode...")

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                learner.learn_pred_V(t, state, action, next_state, reward, done, info)
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V += [learner.getV().getValue(state_observe)]

                if self.debug:
                    print("-> {}".format(self.env.getState()), end=" ")

            if compute_rmse:
                if self.env.getV() is not None:
                    RMSE[episode] = rmse(self.env.getV(), learner.getV().getValues())#, weights=learner.getStateCounts())

            if plot:
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

        # Comment this out to NOT show the plot right away in case the calling function adds a new plot to the graph generated here 
        if plot:
            #plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
            plt.figure(fig_V.number)
            ax = plt.gca()
            ax2 = ax.twinx()    # Create a secondary axis sharing the same x axis
            ax2.bar(self.env.all_states, learner.getStateCounts(), color="blue", alpha=0.3)
            plt.sca(ax) # Go back to the primary axis
            #plt.figure(fig_V.number)

        # Restore the initial state distribution of the environment (for the next simulation)
        # Note that the restore step done in the __exit__() method may NOT be enough, because the Simulator object
        # may still exist once the simulation is over.
        if self._isd_orig is not None:
            self.env.setInitialStateDistribution(self._isd_orig)
            self._isd_orig = None

        return  learner.getV().getValues(), learner.getStateCounts(), RMSE, \
                {# Value of alpha for each state at the end of the LAST episode run
                 'alphas_at_episode_end': learner._alphas,
                 # (Average) alpha by episode (averaged over visited states in the episode)
                 'alphas_by_episode': learner.alpha_mean_by_episode
                 }

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


if __name__ == "__main__":
    import runpy
    runpy.run_path("../../setup.py")

    #from Python.lib import environments, agents
    from environments import gridworlds
    from Python.lib.agents.policies import random_walks
    from Python.lib.agents.learners import mc
    from Python.lib.agents.learners import td

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
