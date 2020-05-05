# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:16:50 2020

@author: Daniel Mastropietro
"""

import numpy as np
import matplotlib.pyplot as plt


class EpisodeSimulation:
    
    def plot_results(self, V_estimated, V_true, RMSE_by_episode, alphas_by_episode, alpha_min,
                     max_rmse=0.8, color_rmse="black", plotFlag=True):
        if plotFlag:
            plt.figure()
            plt.plot(np.arange(self.nS+2), V_true, 'b.-')
            plt.plot(np.arange(self.nS+2), V_estimated, 'r.-')
            plt.title(self.id())
    
            plt.figure()
            plt.plot(np.arange(self.nrounds)+1, RMSE_by_episode, color=color_rmse)
            #plt.xticks(np.arange(self.nrounds)+1)
            ax = plt.gca()
            #ax.set_ylim((0, np.max(RMSE_by_episode)))
            ax.set_ylim((0, max_rmse))
            ax.set_xlabel("Episode")
            ax.set_ylabel("RMSE")
            ax.set_title(self.id())
            
            ax2 = ax.twinx()
            ax2.plot(np.arange(self.nrounds)+1, alphas_by_episode, "k:")
            ax2.set_ylabel("(Average) alpha")
            ax2.axhline(y=alpha_min, color="gray")

    def array2str(self, x, sep=", ", fmt=":.6f"):
        "Converts an array (possibly numeric) to string separated by `sep`"
        return "[" + sep.join( map(lambda s: ("{" + fmt + "}").format(s), x) ) + "]"
