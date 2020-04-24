# -*- coding: utf-8 -*-
"""
Created on Sat Apr  11 15:11:21 2020

@author: Daniel Mastropietro
@description: Write unit tests learning with MC(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
#from gym.utils import seeding
import matplotlib.pyplot as plt

from Python.lib.environments import gridworlds
from Python.lib.agents.policies import random_walks
import Python.lib.agents as agents
from Python.lib.agents.learners import mc
import Python.lib.simulators as simulators

#from importlib import reload
#import Python.lib.agents.learners.mc
#reload(Python.lib.agents.learners.mc)
#from Python.lib.agents.learners.mc import LeaMCLambda


class Test_MC_Lambda(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nrounds = 20
        self.start_state = 9

    @classmethod
    def setUpClass(cls):    # cls is the class
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.plotFlag = True

        cls.nS = 19             # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states
        # True state value functions
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0
        # Agent with Policy and Learner defined
        cls.rw = random_walks.PolRandomWalkDiscrete(cls.env)
        cls.mclambda = mc.LeaMCLambda(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8)
        cls.mclambda_adaptive = mc.LeaMCLambdaAdaptive(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8)
        cls.agent_rw_mclambda = agents.GeneralAgent(cls.rw, cls.mclambda)
        cls.agent_rw_mclambda_adaptive = agents.GeneralAgent(cls.rw, cls.mclambda_adaptive)

    def plot_results(self, V_estimated, V_true, RMSE_by_episode, plotFlag):
        if plotFlag:
            plt.figure()
            plt.plot(np.arange(self.nS+2), V_true, 'b.-')
            plt.plot(np.arange(self.nS+2), V_estimated, 'r.-')
            plt.title(self.id())
            
            plt.figure()
            plt.plot(np.arange(self.nrounds)+1, RMSE_by_episode, color="black")
            plt.xticks(np.arange(self.nrounds)+1)
            ax = plt.gca()
            #ax.set_ylim((0, np.max(RMSE_by_episode)))
            ax.set_ylim((0, 0.5))
            ax.set_xlabel("Episode")
            ax.set_ylabel("RMSE")
            ax.set_title(self.id())

    #------------------------------------------- TESTS ----------------------------------------
    def test_random_walk_result(self):
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode = sim.play(nrounds=self.nrounds, start=9, seed=self.seed, compute_rmse=True, plot=False, pause=0.1)

        expected = np.array([ 0.,     -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
 -0.6319687,  -0.52116762, -0.44295159, -0.20887109, -0.1027944,  -0.03800919,
 -0.03668617,  0.06142266,  0.27410733,  0.42610526,  0.50467228,  0.63018903,
  0.6727829,   0.72310919,  0.        ])
    
        observed = self.mclambda.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode, self.plotFlag)

        assert np.allclose( expected, observed )

    def no_test_random_walk_adaptive_result(self):
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda_adaptive, debug=False)
        _, _, RMSE_by_episode = sim.play(nrounds=self.nrounds, start=9, seed=self.seed, compute_rmse=True, plot=False, pause=0.1)

        expected = np.array([ 0.,     -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
 -0.6319687,  -0.52116762, -0.44295159, -0.20887109, -0.1027944,  -0.03800919,
 -0.03668617,  0.06142266,  0.27410733,  0.42610526,  0.50467228,  0.63018903,
  0.6727829,   0.72310919,  0.        ])
        observed = self.mclambda_adaptive.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode, self.plotFlag)

        assert np.allclose( expected, observed )
    #------------------------------------------- TESTS ----------------------------------------



if __name__ == "__main__":
    unittest.main()
