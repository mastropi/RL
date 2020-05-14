# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:16:50 2020

@author: Daniel Mastropietro
"""

import numpy as np
import matplotlib.pyplot as plt


class EpisodeSimulation:
    
    def plot_results(self, params,
                     V_estimated, V_true, RMSE_by_episode, alphas_by_episode,
                     max_rmse=0.8, color_rmse="black", plotFlag=True):
        """
        Plots the estimated and true state value function.
        
        Arguments:
        params: dict
            Dictionary with simulation parameters:
            - 'alpha': learning rate
            - 'gamma': discount factor of the environment
            - 'lambda': lambda parameter in TD(lambda)
            - 'alpha_min': lower bound for the learning rate used when running the simulation       

        V_estimated: numpy.array of length number of states in the environment
            Estimated state value function.
            
        V_true: numpy.array of length number of states in the environment
            True state value function.

        RMSE_by_episode: numpy.array of length number of episodes run
            Root Mean Squared Error of the estimated state value function by episode.
        
        alphas_by_episode: list or numpy.array
            Average learning rate by episode.
        """
        if plotFlag:
            title = "alpha={:.2f}, gamma={:.2f}, lambda={:.2f}, {} episodes" \
                         .format(params['alpha'], params['gamma'], params['lambda'], self.nrounds)

            plt.figure()
            plt.plot(np.arange(self.nS+2), V_true, 'b.-')
            plt.plot(np.arange(self.nS+2), V_estimated, 'r.-')
            plt.title(title)
    
            plt.figure()
            plt.plot(np.arange(self.nrounds)+1, RMSE_by_episode, color=color_rmse)
            #plt.xticks(np.arange(self.nrounds)+1)
            ax = plt.gca()
            #ax.set_ylim((0, np.max(RMSE_by_episode)))
            ax.set_ylim((0, max_rmse))
            ax.set_xlabel("Episode")
            ax.set_ylabel("RMSE")
            ax.set_title(title)
            
            ax2 = ax.twinx()
            ax2.plot(np.arange(self.nrounds)+1, alphas_by_episode, "k:")
            ax2.set_ylabel("(Average) alpha")
            ax2.axhline(y=params['alpha_min'], color="gray")

    def array2str(self, x, sep=", ", fmt=":.6f"):
        "Converts an array (possibly numeric) to string separated by `sep`"
        return "[" + sep.join( map(lambda s: ("{" + fmt + "}").format(s), x) ) + "]"
