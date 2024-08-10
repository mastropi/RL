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

    @classmethod
    def setUpClass(cls):
        nv = 20
        cls.env_mc = MountainCarDiscrete(nv, factor_for_force_and_gravity=100, debug=True)
        print(f"\nPosition grid values ({len(cls.env_mc.get_positions())} points): {cls.env_mc.get_positions()}")
        print(f"\nVelocity grid values ({len(cls.env_mc.get_velocities())} points): {cls.env_mc.get_velocities()}")

    def test_states(self):
        "Tests the consistency of the different states stored in the object: 2D continuous-valued state (x, v), 1D index, 2D index"
        positions = self.env_mc.get_positions()
        velocities = self.env_mc.get_velocities()
        dict_cases = {# Near-valley state (this is a continuous-valued state and is most likely NOT a point of the discretized grid
                      1: {'state': (-0.5, 0.0), 'idx_state_expected': 118},
                      # Goal state
                      2: {'state': (0.5, 0.0) , 'idx_state_expected': 131},
                      # The state with the smallest 1D index
                      3: {'state': (positions[0], velocities[0]), 'idx_state_expected': 0},
                      # The state with the largest 1D index
                      4: {'state': (positions[-1], velocities[-1]), 'idx_state_expected': self.env_mc.nS - 1}}
        for item in enumerate(dict_cases.items()):
            i = item[0]
            case = item[1][1]
            print(f"\nTesting case {i+1} of {len(dict_cases)}...")
            print(case)
            state = case['state']
            idx_state_expected = case['idx_state_expected']
            idx_state = self.env_mc.get_index_from_state(state)
            state_discrete = self.env_mc.discretize(state)
            idx_state_discrete = self.env_mc.get_index_from_state_discrete(state_discrete)
            print(f"State = {state}:")
            print(f"--> 1D index: {idx_state}")
            print(f"--> 2D index: {state_discrete}")
            print(f"--> 1D index of 2D index: {idx_state_discrete}")
            print(f"--> (x, v) of the 1D index: {self.env_mc.get_state_from_index(idx_state)}")
            #print(f"--> (x, v) of the 2D index: {self.env_mc.get_state_from_discrete_state(state_discrete)}")

            assert idx_state == idx_state_expected
            assert idx_state == idx_state_discrete

    def test_movement(self):
        "Tests the movement of the car, based on the discretization used"
        for position in self.env_mc.get_positions():
            state = (position, 0.0)
            idx_state = self.env_mc.get_index_from_state(state)
            print("\nMoving from state=({:.4f}, {:.4f}) on different accelerations".format(state[0], state[1]))
            self.env_mc.setState(idx_state)

            # Different accelerations
            for action in np.arange(self.env_mc.getNumActions()):
                #idx_next_state, reward, done, info = self.env_mc.step(action)
                #next_state = self.env_mc.get_state_from_index(idx_next_state)
                #print("Moving from state=({:.4f}, {:.4f}) with acceleration a={} => next state: {}".format(state[0], state[1], action-1, next_state))
                next_state_cont, next_state_discrete, reward, done, info = self.env_mc.step(action, return_continuous_state=True)
                next_state = self.env_mc.get_state_from_discrete_state(next_state_discrete)
                print("Moving from state=({:.4f}, {:.4f}) with acceleration a={} => next state: (x, v) = {}, (xd, vd) = {}".format(state[0], state[1], action-1, next_state_cont, next_state))
                if np.isclose(next_state[0], state[0]):
                    print("---> WARNING: The car's position did NOT change!")
                # Reset the state to the one before the action was taken
                self.env_mc.setState(idx_state)
                # Check that we went back to the original state before moving
                assert np.allclose(self.env_mc.get_state_from_index( self.env_mc.getState() ), state)

    def no_test_get_from_adjacent_states(self):
        print("\nRunning test {}...".format(self.id()))

        # Number of discrete velocities
        nv = 20
        env_mc = MountainCarDiscrete(nv, factor_for_force_and_gravity=20, debug=True)
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
    #test_suite.addTest(Test_Support_EnvMountainCars("test_method_plot"))
    test_suite.addTest(Test_Support_EnvMountainCars("test_movement"))
    runner.run(test_suite)
