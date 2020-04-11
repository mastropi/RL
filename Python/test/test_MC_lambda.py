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
        cls.mc_lambda = mc.LeaMCLambda(cls.env, alpha=0.2, gamma=0.9, lmbda=0.8)
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
    def test_random_walk_result(self):
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
