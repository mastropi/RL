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
    from agents import GeneralAgent
else:
    # These relative imports are only accepted when we compile the file as a module
    from .environments import EnvironmentDiscrete
    from .agents import GeneralAgent


class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a given environment `env`
    and an `agent`.
    """

    def __init__(self, env, agent, seed=None, debug=False):
        if not isinstance(env, EnvironmentDiscrete):
            raise TypeError("The environment must be of type {} from the {} module ({})" \
                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        # TODO: (2020/04/12) Fix the following check on the type of agent as it does not work...
        # even though the class of agent is Python.lib.agents.GeneralAgent
#        if not isinstance(agent, GeneralAgent):
#            raise TypeError("The agent must be of type {} from the {} module ({})" \
#                            .format(GeneralAgent.__name__, GeneralAgent.__module__, agent.__class__))

        self.debug = debug

        self.env = env
        self.agent = agent
        self.seed = seed

        self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the initial state distribution array in case it has been possibly changed
        # (e.g. when we want to define a specific initial state)
        if self._isd_orig:
            self.setInitialStateDistribution(self._isd_orig)

    def reset(self):
        # Copy of Initial State Distribution of environment in case we need to change it
        self._isd_orig = None

        # Reset the state counts in an episode
        # (This is used for plotting purposes and to compute the RMSE weighted by the state counts --if requested)
        self.state_counts = np.zeros(self.env.getNumStates())

        # Reset the learner to the first episode state
        self.agent.getLearner().reset(reset_episode=True, reset_value_functions=True)

    def play(self, nrounds, start=None, seed=None, compute_rmse=False, plot=False, colormap="seismic", pause=0):
        # TODO: (2020/04/11) Convert the plotting parameters to a dictionary named plot_options or similar.
        # The goal is to group OPTIONAL parameters by their function/concept.  
        """Runs an episodic Reinforcement Learning experiment.
        No reset of the learning history is done at the onset of the first episode to run.
        It is assumed that this reset, when needed, is done by the calling function.
        
        Parameters:
        nrounds: int
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
            Tuple containing the following elements having the same size as the environment states:
                - state value function estimates
                - number of visits at each state
                - RMSE (when compute_rmse is not None) an array of length `nrounds` containing the
                Root Mean Square Error after each episode, of the estimated value function averaged
                over all states and weighted by the number of visits to each state. Otherwise, None.
        """
        if plot:
            fig_V = plt.figure()
            colors = cm.get_cmap(colormap, lut=nrounds)
            # Plot the true state value function (to have it as a reference already
            plt.plot(self.env.all_states, self.env.getV(), '.-', color="blue")

        # Define initial state
        if start:
            nS = self.env.getNumStates()
            if not (isinstance(start, int) and 0 <= start and start < nS):
                raise Warning('The `start` parameter must be an integer number between 0 and {}.' \
                              'A start state will be selected based on the initial state distribution of the environment.'.format(nS-1))
            else:
                # Change the initial state distribution of the environment so that
                # the environment resets to start at the given 'start' state.
                self._isd_orig = self.env.getInitialStateDistribution()
                isd = np.zeros(nS)
                isd[start] = 1.0
                self.env.setInitialStateDistribution(isd)

        # Define the policy, the learner and reset the learner (i.e. erase all learning memory to start anew!)
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()
        if seed:
            if seed != 0:
                self.env.seed(seed)
            else:
                self.env.seed(self.seed)

        RMSE = np.nan*np.zeros(nrounds) if compute_rmse else None
        for episode in range(nrounds):
            self.env.reset()
            learner.reset(reset_episode=False, reset_value_functions=False)
            done = False
            if self.debug:
                print("\n\nEpisode {} starts...".format(episode+1))
                print("\nStarts at state {}".format(self.env.getState()))
                print("\tState value function at start of episode:\n\t{}".format(learner.getV().getValues()))

            # Time step in the episode (the first time step is t = 0
            t = -1
            while not done:
                t += 1
                
                # Current state and action on that state leading to the next state
                state = self.env.getState()
                self.state_counts[state] += 1
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

                if self.debug:
                    print("-> {}".format(self.env.getState()), end=" ")

            if compute_rmse:
                if self.env.getV() is not None:
                    RMSE[episode] = rmse(self.env.getV(), learner.getV().getValues(), weights=self.state_counts)

            if plot:
                #print("episode: {} (T={}), color: {}".format(episode, t, colors(episode/nrounds)))
                plt.figure(fig_V.number)
                plt.plot(self.env.all_states, learner.getV().getValues(), linewidth=0.5, color=colors(episode/nrounds))
                if pause > 0:
                    plt.pause(pause)
                plt.draw()
                #fig.canvas.draw()

        # Comment this out to NOT show the plot rightaway in case the calling function adds a new plot to the graph generated here 
        if plot:
            #plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
            ax = plt.gca()
            ax2 = ax.twinx()    # Create a secondary axis sharing the same x axis
            ax2.bar(self.env.all_states, self.state_counts, color="blue", alpha=0.3)
            plt.sca(ax) # Go back to the primary axis
            #plt.figure(fig_V.number)

        # TODO: DO WE NEED THIS RESTORE OF THE INITIAL STATE DISTRIBUTION OR THE __exit__() METHOD SUFFICES??
#        if self.isd_orig:
#            self.env.setInitialStateDistribution(self.isd_orig)

        return learner.getV().getValues(), self.state_counts, RMSE

    def simulate(self, nexperiments, nepisodes, start=None, verbose=False):
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

        Returns: tuple
            Tuple containing the following elements:
                Avg(RMSE): Root Mean Square Error averaged over all experiments
                SE(RMSE): Standard Error of the average RMSE
                Episodic RMSE: array containing the Root Mean Square Error by episode averaged
                over all experiments.
        """
    
        if not (isinstance(nexperiments, int) and nexperiments > 0):
            raise ValueError("The number of experiments must be a positive integer number ({})".format(nexperiments))
        if not (isinstance(nepisodes, int) and nepisodes > 0):
            raise ValueError("The number of episodes must be a positive integer number ({})".format(nepisodes))

        RMSE = 0.
        RMSE2 = 0.  # Used to compute the standard error of the average RMSE
        RMSE_by_episodes = np.zeros(nepisodes)
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
            self.agent.getLearner().reset(reset_episode=True, reset_value_functions=True)
            V, N, RMSE_by_episodes_i = self.play(nrounds=nepisodes, start=start, seed=None,
                                                 compute_rmse=True, plot=False)
            RMSE_i = rmse(self.env.getV(), V) #, weights=N)
            RMSE += RMSE_i
            RMSE2 += RMSE_i**2
            RMSE_by_episodes += RMSE_by_episodes_i
            RMSE_by_episodes2 += RMSE_by_episodes_i**2
            if verbose:
                print("\tRMSE(experiment) = {:.3g}".format(RMSE_i))

        RMSE_mean = RMSE / nexperiments
        RMSE_se = np.sqrt( ( RMSE2 - nexperiments*RMSE_mean**2 ) / (nexperiments - 1) ) \
                / np.sqrt(nexperiments)
            ## The above, before taking sqrt is the estimator of the Var(RMSE) = (SS - n*mean^2) / (n-1)))
        RMSE_by_episodes_mean = RMSE_by_episodes / nexperiments
        RMSE_by_episodes_se = np.sqrt( ( RMSE_by_episodes2 - nexperiments*RMSE_by_episodes_mean**2 ) / (nexperiments - 1) ) \
                            / np.sqrt(nexperiments)

        return RMSE_mean, RMSE_se, RMSE_by_episodes_mean, RMSE_by_episodes_se


# TODO: (2020/04/11) Move this function to an util module
def rmse(Vtrue, Vest, weights=None):
    """Root Mean Square Error (RMSE) between Vtrue and Vest, weighted or not weighted by weights.
    @param Vtrue: True Value function.
    @param Vest: Estimated value function.
    @param weights: Number of visits for each value.
    """

    assert type(Vtrue) == np.ndarray and type(Vest) == np.ndarray and (weights is None or type(weights) == np.ndarray), \
            "The first three input parameters are numpy arrays"
    assert Vtrue.shape == Vest.shape and (weights is None or Vest.shape == weights.shape), \
            "The first three input parameters have the same shape({}, {}, {})" \
            .format(Vtrue.shape, Vest.shape, weights and weights.shape or "")

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    if weights is not None:
        mse = np.sum( weights * (Vtrue - Vest)**2 ) / np.sum(weights)
    else:
        mse = np.mean( (Vtrue - Vest)**2 )

    return np.sqrt(mse)


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
    fig, (ax_full, ax_scaled, ax_rmse_by_episode) = plt.subplots(1,3)

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
    nexperiments = 2
    nepisodes = 5
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
        alphas = np.linspace(0.1, 1.2, 5)
        #lambdas = [0.99]
        #alphas = [0.41]
        n_lambdas = len(lambdas)
        n_alphas = len(alphas)
    n_simul = n_lambdas*n_alphas

    #learner.setParams(alpha=0.41, gamma=gamma, lmbda=0.1)
    #agent = GeneralAgent(pol_rw,
    #                     learner)
    #sim = Simulator(env, agent)
    #rmse_mean, rmse_se = sim.simulate(nexperiments=nexperiments, nepisodes=nepisodes, verbose=verbose)
    #print(rmse_mean, rmse_se)

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
            agent = GeneralAgent(pol_rw,
                                 learner)
            # NOTE: Setting the seed here implies that each set of experiments
            # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
            # of visited states and actions.
            # This is DESIRED --as opposed of having different state-action outcomes for different
            # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
            # VERIFIED BY RUNNING IN DEBUG MODE!
            sim = Simulator(env, agent, seed=seed, debug=debug)

            # Run the simulation and store the results
            rmse_mean, rmse_se, rmse_episodes = sim.simulate(nexperiments=nexperiments,
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
        color = colormap( idx_lmbda / np.max((1, n_lambdas-1)) )
        ax_full.plot(alphas, rmse_mean_lambda, '.', color=color)
        ax_full.errorbar(alphas, rmse_mean_lambda, yerr=rmse_se_lambda, capsize=4, color=color)
        legend_label += ["lam={:.2g}".format(lmbda)]

    # Average RMSE by episode for convergence analysis
    rmse_episodes_values /= n_simul

    # JUST ONE FINAL RUN
    #learner.setParams(alpha=0.3, gamma=gamma, lmbda=1)
    #agent = GeneralAgent(pol_rw,
    #                     learner)
    #sim = Simulator(env, agent)
    #rmse_mean, rmse_se = sim.simulate(nexperiments=nexperiments, nepisodes=nepisodes, verbose=verbose)
    #if verbose:
    #    print("\tRMSE = {:.3g} ({:.3g})".format(rmse_mean, rmse_se))

    # Scaled plot (for comparison purposes)
    for idx_lmbda, lmbda in enumerate(lambdas):
        #rmse2plot = rmse_mean_values[idx_lmbda]
        #rmse2plot_error = rmse_se_values[idx_lmbda]
        rmse2plot = rmse_episodes_mean[idx_lmbda]
        rmse2plot_error = rmse_episodes_se[idx_lmbda]
        color = colormap( 1 - idx_lmbda / np.max((1, n_lambdas-1)) )
        ax_scaled.plot(alphas, rmse2plot, '.-', color=color)
        ax_scaled.errorbar(alphas, rmse2plot, yerr=rmse2plot_error, capsize=4, color=color)
        ax_scaled.set_xlim((0, max_alpha))
        ax_scaled.set_ylim((0, max_rmse))

    # Episodic RMSE
    ax_rmse_by_episode.plot(np.arange(nepisodes), rmse_episodes_values, color="black")
    ax_rmse_by_episode.set_ylim((0, max_rmse))
    ax_rmse_by_episode.set_xlabel("Episode")
    ax_rmse_by_episode.set_ylabel("RMSE")
    ax_rmse_by_episode.set_title("Average RMSE by episode over ALL experiments")

    plt.figlegend(legend_label)
    fig.suptitle("{}: gamma = {:.2g}, #experiments = {}, #episodes = {}"\
                 .format(learner.__class__, gamma, nexperiments, nepisodes))
