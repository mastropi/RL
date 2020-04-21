# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:40:41 2020

@author: Daniel Mastropietro
@description: Write unit tests for the gridworld learning using TD(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
#from gym.utils import seeding
import matplotlib.pyplot as plt

from Python.lib.environments import gridworlds
import Python.lib.agents as agents
from Python.lib.agents.policies import random_walks
from Python.lib.agents.learners import td
import Python.lib.simulators as simulators

#from importlib import reload
#import Python.lib.agents.learners.td
#reload(Python.lib.agents.learners.td)
#from Python.lib.agents.learners.td import LeaTDLambda


class Test_TD_Lambda(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nrounds = 20
        self.start_state = 9

    @classmethod
    def setUpClass(cls):    # cls is the class, in this case, class 'Test_TD_Lambda'
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.plotFlag = True

        cls.nS = 19             # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states
        # True state value functions
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0
        # Environment
        cls.env = gridworlds.EnvGridworld1D(length=21)  # 19 states plus the two terminal states
        # Agents with Policy and Learner defined
        cls.rw = random_walks.PolRandomWalkDiscrete(cls.env)
        cls.tdlambda = td.LeaTDLambda(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8)
        cls.tdlambda_adaptive = td.LeaTDLambdaAdaptive(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8)
        cls.mclambda = td.LeaTDLambda(cls.env, alpha=0.2, gamma=0.9, lmbda=1.0)
        cls.agent_rw_tdlambda = agents.GeneralAgent(cls.rw, cls.tdlambda)
        cls.agent_rw_tdlambda_adaptive = agents.GeneralAgent(cls.rw, cls.tdlambda_adaptive)
        cls.agent_rw_mclambda = agents.GeneralAgent(cls.rw, cls.mclambda)

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
        "Test using my TD(Lambda) learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode = sim.play(nrounds=self.nrounds, start=9, seed=self.seed, compute_rmse=True, plot=False)

        expected = np.array([ 0.,       -0.90853729, -0.87876371, -0.70629267, -0.61659491, -0.62814336,
 -0.59673825, -0.49258608, -0.40412844, -0.21597426, -0.15834096, -0.10753559,
 -0.10549221,  0.01441717,  0.28382445,  0.47064362,  0.47555858,  0.57177841,
  0.62845292,  0.70442683,  0.        ])
        observed = self.tdlambda.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode, self.plotFlag)

        assert np.allclose( expected, observed )

    def test_random_walk_as_MC_result(self):
        "Test using my TD(Lambda) with lambda = 1 to emulate a Monte Carlo learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode = sim.play(nrounds=self.nrounds, start=9, seed=self.seed, compute_rmse=True, plot=False)

        expected = np.array([ 0.,        -0.8598213, -0.76779645, -0.37056923, -0.18913217, -0.23620264,
 -0.18414855, -0.12397922, -0.1037322,  -0.03250992, -0.02509062, -0.04620496,
 -0.06607403, -0.00726447,  0.2325766,   0.35186274,  0.35691905,  0.50138679,
  0.4908309,   0.63692378,  0.        ])
        observed = self.mclambda.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode, self.plotFlag)

        assert np.allclose( expected, observed )

    def test_random_walk_adaptive_result(self):
        "Test using my TD(Lambda) learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode = sim.play(nrounds=self.nrounds, start=9, seed=self.seed, compute_rmse=True, plot=False)

        expected = np.array([ 0.,       -0.83763831, -0.78371244, -0.65037339, -0.55487832, -0.45724707,
 -0.36835343, -0.25302472, -0.2056898,  -0.09006464, -0.05197223,  0.01445287,
  0.04177646,  0.09774879,  0.16473479,  0.22615434,  0.35857778,  0.44099474,
  0.56933659,  0.71409178,  0.        ])
        observed = self.tdlambda_adaptive.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode, self.plotFlag)

        assert np.allclose( expected, observed )
    #------------------------------------------- TESTS ----------------------------------------



if __name__ == "__main__":
    unittest.main()
