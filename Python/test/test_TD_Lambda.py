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

from Python.lib.GitHub.MJeremy2017.RL.RandomWalk_Lambda import TD_Lambda as TDL
from Python.lib import TD_Lambda_DM as TDLDM
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

        # Setup based on the code from MJeremy2017 (which was a little modified)
        cls.rw_MJDM = TDLDM.RandomWalk_DM(start=9, lmbda=0.8, debug=False)
        cls.rw_td_MJDM = TDL.RWTD(start=9, debug=False)

        # Using the MJeremy2017 definition refactored into separate concepts (env, policies, learners)
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

    def setUp(self):
        # Make the tests repeatable
        # NOTE: This setUp() is run before EVERY test, as there is setUpClass() that is run
        # just once before ALL the tests are run.
        #self.random_generator, seed = seeding.np_random(self.seed)
        # This is used by the MJeremy2017 implementation (since they call np.random.choice() to choose the action)
        # (as opposed to seeding.np_random() which is the case in the DiscreteEnv environment of gym --which I use)
        np.random.seed(self.seed)

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
    def test_random_walk_result_MJDM(self):
        "Test using the off-line learning (that calls gt2tn() implemented in TD_Lambda_DM.py"
        print("\nTesting " + self.id())
        valueFunc = TDLDM.ValueFunction_DM(alpha=0.2)
        self.rw_MJDM.play(valueFunc, rounds=self.nrounds)

        expected = np.array([ 0.,         -0.97026362, -1.09293459, -0.9594995,  -0.8038009,  -0.74118132,
 -0.70110327, -0.5745329,  -0.38799777, -0.12274256, -0.03652859,  0.16429707,
  0.37958913,  0.43674974,  0.46251896,  0.60353857,  0.83647189,  0.94198936,
  0.95498279,  0.84075131,  0.        ])
# DM-2020/04/06: Previous result, when an action was spared at the end of each episode because
# it was taken BEFORE going to the next iteration (as opposed to take it soon after the new iteration starts)
#                [ 0.,         -0.99498334, -1.12844023, -1.08742679, -0.99925887, -0.92014644,
# -0.83281915, -0.73404135, -0.67868899, -0.54605007, -0.32858876,  0.09982778,
#  0.14763525,  0.19785301,  0.33526039,  0.59847646,  0.66730606,  0.86347524,
#  0.99748532,  0.95018769,  0.        ])
        observed = valueFunc.getValues()

        print("\nobserved: " + str(observed))

        assert np.allclose( expected, observed )

    def test_random_walk_td_result_MJDM(self):
        "Test using MJeremy2017' TD(Lambda) learner"
        print("\nTesting " + self.id())
        valueFunc = TDLDM.ValueFunctionTD(alpha=0.2, gamma=0.9, lmbda=0.8)
        self.rw_td_MJDM.play(valueFunc, rounds=self.nrounds)

        expected = np.array([ 0.,         -0.93600092, -0.62865005, -0.32600362, -0.23004461, -0.20544315,
 -0.17822048, -0.1183389,  -0.04995117, -0.01169715, -0.00284312,  0.02066435,
  0.05085406,  0.05865383, 0.07223799,  0.15062597,  0.32170616,  0.43413232,
  0.49881925,  0.73350279, 0.        ])
        observed = valueFunc.getValues()

        print("\nobserved: " + str(observed))

        assert np.allclose( expected, observed )

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
