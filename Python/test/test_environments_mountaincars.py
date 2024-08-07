# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:18:51 2024

@author: Alphonse Lafon
@description: Test of the mountain car environments.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest

import numpy as np
import pandas as pd

from Python.lib.environments.mountaincars import MountainCarDiscrete


class Test_Support_EnvMountainCars(unittest.TestCase):

    def test_get_from_adjacent_states(self):
        print("\nRunning test {}...".format(self.id()))

        # Number of discrete velocities
        nv = 20
        env_mc = MountainCarDiscrete(nv, debug=True)
        print(f"\nTotal number of 2D states in the discrete Mountain Car: {len(env_mc.get_positions())} positions x {len(env_mc.get_velocities())} velocities = {env_mc.getNumStates()}")
        print("\nPossible discrete positions of the Mountain Car: (dx={:.4f})".format(env_mc.dx))
        print(pd.DataFrame({'idx_x': np.arange(len(env_mc.get_positions())),
                            'x': env_mc.get_positions()}, columns=['idx_x', 'x']))
        print("Possible discrete velocities of the Mountain Car: (dv={:.4f})".format(env_mc.dv))
        print(pd.DataFrame({'idx_v': np.arange(len(env_mc.get_velocities())),
                            'v': env_mc.get_velocities()}, columns=['idx_v', 'v']))

        state_list = [(-0.7, -0.06),      # To the left of the valley going left
                      (-0.5,  0.00),      # Approximately the valley of the path
                      ( 0.0,  0.00),      # Somewhere on the right of the valley with zero velocity (this is actually not of interest because the car will never end up here)
                      ( 0.2, -0.03),      # Somewhere on the right with negative velocity
                      ( 0.5,  0.05)]      # The finish line with positive velocity
        expected_list = [[(-0.6356666666666666, -0.05600000000000001)],
                         [(-0.5066761904761905, -0.013999999999999999), (-0.5066761904761905, 0.007000000000000006), (-0.5066761904761905, 0.02800000000000001)],
                         [(-0.00683809523809531, 0.035), (-0.00683809523809531, 0.05600000000000002), (-0.00683809523809531, 0.07)],
                         [(0.21889523809523803, 0.02800000000000001)],
                         [(0.4607523809523808, 0.04200000000000001), (0.4607523809523808, 0.063), (0.47687619047619045, 0.04200000000000001), (0.47687619047619045, 0.05600000000000002), (0.493, 0.035), (0.493, 0.05600000000000002), (0.5, 0.035), (0.5, 0.05600000000000002)]]
        for i, (state, expected) in enumerate(zip(state_list, expected_list)):
            print(f"\nTest state #{i}: {state}")
            observed_set_of_adjacent_states = env_mc.get_from_adjacent_states(state)
            print(f"Set of adjacent states: {observed_set_of_adjacent_states}")
            assert observed_set_of_adjacent_states == expected
            # Check that the adjacent states are among the set of discretized (x, v) states
            for (x, v) in observed_set_of_adjacent_states:
                assert x in env_mc.get_positions()
                assert v in env_mc.get_velocities()

    def test_method_plot(self):
        print("\nRunning test {}...".format(self.id()))
        nv = 20
        env_mc = MountainCarDiscrete(nv)
        state_list = [
            [-0.5, 0.1],
            [-0.45, 0.1],
            [-0.4, 0.1],
            [-0.35, 0.1],
            [-0.3, 0.1],
            [-0.25, 0.2],
            [-0.2, 0.2],
            [-0.15, 0.2],
            [-0.1, 0.2],
            [-0.05, 0.2],
            [0.0, 0.2],
            [0.05, 0.2],
            [0.1, 0.3],
            [0.15, 0.3],
            [0.2, 0.3],
            [0.3, 0.3],
            [0.4, 0.3],
            [0.5, 0.3],
            [0.45, -0.1],
            [0.35, -0.1],
            [0.2, -0.1],
            [0.05, 0.0]]
        env_mc.plot(state_list)


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    #unittest.main()

    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_Support_EnvMountainCars("test_get_from_adjacent_states"))
    #test_suite.addTest(Test_Support_EnvMountainCars("test_method_plot"))
    runner.run(test_suite)
