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
        cls.td_lambda = td.LeaTDLambda(cls.env, alpha=0.2, gamma=0.9, lmbda=0.8)
        cls.mc_lambda = td.LeaTDLambda(cls.env, alpha=0.2, gamma=0.9, lmbda=1.0)
        cls.agent_td_lambda = agents.PolicyLearner(cls.rw, cls.td_lambda)
        cls.agent_mc_lambda = agents.PolicyLearner(cls.rw, cls.mc_lambda)
    
    def setUp(self):
        # Make the tests repeatable
        # NOTE: This setUp() is run before EVERY test, as there is setUpClass() that is run
        # just once before ALL the tests are run.
        #self.random_generator, seed = seeding.np_random(self.seed)
        np.random.seed(self.seed)

    def plot_results(self, V_estimated, V_true, plotFlag):
        if plotFlag:
            plt.plot(np.arange(self.nS+2), V_true, 'b.-')
            plt.plot(np.arange(self.nS+2), V_estimated, 'r.-')
            plt.title(self.id())
            plt.show()

    #------------------------------------------- TESTS ----------------------------------------
    def no_test_random_walk_result_MJDM(self):
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
        sim = simulators.Simulator(self.env, self.agent_td_lambda, debug=False)
        sim.play(start=9, nrounds=self.nrounds)

        expected = np.array([ 0.,         -0.93600092, -0.62865005, -0.32600362, -0.23004461, -0.20544315,
 -0.17822048, -0.1183389,  -0.04995117, -0.01169715, -0.00284312,  0.02066435,
  0.05085406,  0.05865383, 0.07223799,  0.15062597,  0.32170616,  0.43413232,
  0.49881925,  0.73350279, 0.        ])
        observed = self.td_lambda.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, self.plotFlag)

        assert np.allclose( expected, observed )

    def test_random_walk_as_MC_result(self):
        "Test using my TD(Lambda) with lambda = 1 to emulate a Monte Carlo learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_mc_lambda, debug=False)
        sim.play(start=9, nrounds=self.nrounds)

        expected = np.array([ 0.,         -0.92187619, -0.62423707, -0.2309414,  -0.138025,  -0.24758464,
 -0.30398583, -0.23128573, -0.08455704, -0.08961004, -0.16759842, -0.07701461,
  0.02173501,  0.01127705,  0.05074384,  0.24794607,  0.48046892,  0.62277033,
  0.65774705,  0.772826,    0.        ])
        observed = self.mc_lambda.getV().getValues()

        print("\nobserved: " + str(observed))

        self.plot_results(observed, self.V_true, self.plotFlag)

        assert np.allclose( expected, observed )
    #------------------------------------------- TESTS ----------------------------------------



if __name__ == "__main__":
    unittest.main()
