# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 18:20:57

@author: Daniel Mastropietro
@description: Unit tests for environments.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from unittest_data_provider import data_provider

import numpy as np

from Python.lib.environments.gridworlds import Direction2D, EnvGridworld2D, get_adjacent_states, get_opposite_direction


class Test_Support_EnvGridworld2D_WithObstacles(unittest.TestCase):

    gridworld = None

    @classmethod
    def setUpClass(cls):
        cls.gridworld = EnvGridworld2D(shape=[3, 4], terminal_states=set({3, 7}), obstacle_states=set({5, 6}), rewards_dict=dict({3: +1, 7: -1}))
        # Set the initial state so that we can test the environment rendering
        cls.gridworld.setState(2)

    def test_show_gridworld(self):
        print("\nRunning test {}...".format(self.id()))
        observed_gridworld = self.gridworld._render(mode='ansi')    # Using mode='ansi' allows for a return value that can be printed here, instead of shown by the render method
        # The expected result is the following but we cannot assert it because the _render() method
        expected_gridworld = """( 0) o  ( 1) o  ( 2) x  ( 3) T
( 4) o  ( 5) P  ( 6) P  ( 7) T
( 8) o  ( 9) o  (10) o  (11) o\n"""
        print(observed_gridworld.getvalue())
        print(expected_gridworld)
        assert observed_gridworld.getvalue() == expected_gridworld

    def test_transition_probabilities(self):
        print("\nRunning test {}...".format(self.id()))
        P_expected = dict({ 0: {Direction2D.UP.value: [(1.0, 0, 0, False)], Direction2D.RIGHT.value: [(1.0, 1, 0, False)], Direction2D.DOWN.value: [(1.0, 4, 0, False)], Direction2D.LEFT.value: [(1.0, 0, 0, False)]},
                            1: {Direction2D.UP.value: [(1.0, 1, 0, False)], Direction2D.RIGHT.value: [(1.0, 2, 0, False)], Direction2D.DOWN.value: [(1.0, 1, 0, False)], Direction2D.LEFT.value: [(1.0, 0, 0, False)]},
                            2: {Direction2D.UP.value: [(1.0, 2, 0, False)], Direction2D.RIGHT.value: [(1.0, 3, +1, True)], Direction2D.DOWN.value: [(1.0, 2, 0, False)], Direction2D.LEFT.value: [(1.0, 1, 0, False)]},
                            3: {Direction2D.UP.value: [(1.0, 3, +1, True)], Direction2D.RIGHT.value: [(1.0, 3, +1, True)], Direction2D.DOWN.value: [(1.0, 3, +1, True)], Direction2D.LEFT.value: [(1.0, 3, +1, True)]},

                            4: {Direction2D.UP.value: [(1.0, 0, 0, False)], Direction2D.RIGHT.value: [(1.0, 4, 0, False)], Direction2D.DOWN.value: [(1.0, 8, 0, False)], Direction2D.LEFT.value: [(1.0, 4, 0, False)]},
                            5: {Direction2D.UP.value: [(1.0, 5, 0, False)], Direction2D.RIGHT.value: [(1.0, 5, 0, False)], Direction2D.DOWN.value: [(1.0, 5, 0, False)], Direction2D.LEFT.value: [(1.0, 5, 0, False)]},
                            6: {Direction2D.UP.value: [(1.0, 6, 0, False)], Direction2D.RIGHT.value: [(1.0, 6, 0, False)], Direction2D.DOWN.value: [(1.0, 6, 0, False)], Direction2D.LEFT.value: [(1.0, 6, 0, False)]},
                            7: {Direction2D.UP.value: [(1.0, 7, -1, True)], Direction2D.RIGHT.value: [(1.0, 7, -1, True)], Direction2D.DOWN.value: [(1.0, 7, -1, True)], Direction2D.LEFT.value: [(1.0, 7, -1, True)]},

                            8: {Direction2D.UP.value: [(1.0, 4, 0, False)], Direction2D.RIGHT.value: [(1.0, 9, 0, False)], Direction2D.DOWN.value: [(1.0, 8, 0, False)], Direction2D.LEFT.value: [(1.0, 8, 0, False)]},
                            9: {Direction2D.UP.value: [(1.0, 9, 0, False)], Direction2D.RIGHT.value: [(1.0, 10, 0, False)], Direction2D.DOWN.value: [(1.0, 9, 0, False)], Direction2D.LEFT.value: [(1.0, 8, 0, False)]},
                            10: {Direction2D.UP.value: [(1.0, 10, 0, False)], Direction2D.RIGHT.value: [(1.0, 11, 0, False)], Direction2D.DOWN.value: [(1.0, 10, 0, False)], Direction2D.LEFT.value: [(1.0, 9, 0, False)]},
                            11: {Direction2D.UP.value: [(1.0, 7, -1, True)], Direction2D.RIGHT.value: [(1.0, 11, 0, False)], Direction2D.DOWN.value: [(1.0, 11, 0, False)], Direction2D.LEFT.value: [(1.0, 10, 0, False)]}})
        print(f"Transition probabilities: {self.gridworld.P}")
        print(f"Transition probabilities (EXPECTED): {P_expected}")
        assert all( np.unique([len(self.gridworld.P[k]) for k in self.gridworld.P.keys()]) == [4] )
        assert self.gridworld.P == P_expected

    def test_method_get_adjacent_states(self):
        print("\nRunning test {}...".format(self.id()))
        # State for which all adjacent states exist
        # Note that this state CAN be an obstacle. In fact, we use this get_adjacent_states() function when constructing
        # the transition probabilities of states that are adjacent to the obstacle!
        state = 5
        observed_set_of_adjacent_states = get_adjacent_states(self.gridworld.getShape(), state)
        expected_set_of_adjacent_states = set([(6, Direction2D.RIGHT), (4, Direction2D.LEFT), (1, Direction2D.UP), (9, Direction2D.DOWN)])
        print(f"Set of adjacent states to s={state} and their directions w.r.t. to s: {observed_set_of_adjacent_states}")
        assert observed_set_of_adjacent_states == expected_set_of_adjacent_states

        # Corner state (for which not all adjacent states exist)
        state = 0
        observed_set_of_adjacent_states = get_adjacent_states(self.gridworld.getShape(), state)
        expected_set_of_adjacent_states = set([(None, Direction2D.UP), (None, Direction2D.LEFT), (1, Direction2D.RIGHT), (4, Direction2D.DOWN)])
        print(f"Set of adjacent states to s={state} and their directions w.r.t. to s: {observed_set_of_adjacent_states}")
        assert observed_set_of_adjacent_states == expected_set_of_adjacent_states

        # Corner state (for which not all adjacent states exist)
        state = 11
        observed_set_of_adjacent_states = get_adjacent_states(self.gridworld.getShape(), state)
        expected_set_of_adjacent_states = set([(None, Direction2D.DOWN), (None, Direction2D.RIGHT), (10, Direction2D.LEFT), (7, Direction2D.UP)])
        print(f"Set of adjacent states to s={state} and their directions w.r.t. to s: {observed_set_of_adjacent_states}")
        assert observed_set_of_adjacent_states == expected_set_of_adjacent_states

        # Border but not corner state (for which not all adjacent states exist)
        state = 1
        observed_set_of_adjacent_states = get_adjacent_states(self.gridworld.getShape(), state)
        expected_set_of_adjacent_states = set([(None, Direction2D.UP), (0, Direction2D.LEFT), (2, Direction2D.RIGHT), (5, Direction2D.DOWN)])
        print(f"Set of adjacent states to s={state} and their directions w.r.t. to s: {observed_set_of_adjacent_states}")
        assert observed_set_of_adjacent_states == expected_set_of_adjacent_states

    def test_method_get_opposite_direction(self):
        print("\nRunning test {}...".format(self.id()))
        direction = Direction2D.UP
        observed_direction = get_opposite_direction(direction)
        expected_direction = Direction2D.DOWN
        print(f"Observed opposite direction to {direction}: {observed_direction}")
        assert observed_direction == expected_direction


class Test_Support_EnvGridworld2D_WithObstaclesAndWind(unittest.TestCase):

    gridworld = None

    @classmethod
    def setUpClass(cls):
        cls.gridworld_up = EnvGridworld2D(shape=[3, 4], terminal_states=set({3, 7}), obstacle_states=set({5}), rewards_dict=dict({3: +1, 7: -1}),
                                       # We use a wind intensity of 1.0 so that we would always move in the direction of the wind and it is easier to test
                                       wind_dict=dict({'direction': Direction2D.UP, 'intensity': 1.0}))
        cls.gridworld_left = EnvGridworld2D(shape=[3, 4], terminal_states=set({3, 7}), obstacle_states=set({5, 6}), rewards_dict=dict({3: +1, 7: -1}),
                                       # We use a wind intensity of 1.0 so that we would always move in the direction of the wind and it is easier to test
                                       wind_dict=dict({'direction': Direction2D.LEFT, 'intensity': 1.0}))

    def test_move(self):
        print("\nRunning test {}...".format(self.id()))

        #--- WIND UP
        env = self.gridworld_up
        
        # Moving left from the left border should stay in the same place (as opposed to moving up because of the wind)
        env.setState(4)
        next_state, reward, _, _ = env.step(Direction2D.LEFT.value)
        assert next_state == 4
        assert env.getState() == next_state
        assert reward == 0

        # Moving right should move up when no obstacles are present
        env.setState(6)
        next_state, reward, _, _ = env.step(Direction2D.RIGHT.value)
        assert next_state == 3
        assert env.getState() == next_state
        assert reward == 1

        # Moving right should NOT move up when there are obstacles in either the cell to the right or the cell where it would move due to the wind
        env.setState(4)
        next_state, reward, _, _ = env.step(Direction2D.DOWN.value)
        assert next_state == 4  # There is an obstacle in state 5, and although the wind would tell the agent to end up in state 1, it can't because there is an obstacle just right from the current state at 4.
        assert env.getState() == next_state
        assert reward == 0

        # Moving up should move just one cell up when the wind ALSO goes up, as opposed to moving *two* cells up
        env.setState(11)
        next_state, reward, _, _ = env.step(Direction2D.UP.value)
        assert next_state == 7
        assert env.getState() == next_state
        assert reward == -1

        env.setState(6)
        next_state, reward, _, _ = env.step(Direction2D.UP.value)
        assert next_state == 2
        assert env.getState() == next_state
        assert reward == 0

        #--- WIND TO THE LEFT
        env = self.gridworld_left

        # Moving up from the top border should stay in the same place (as opposed to moving left because of the wind)
        env.setState(2)
        next_state, reward, _, _ = env.step(Direction2D.UP.value)
        assert next_state == 2
        assert env.getState() == next_state
        assert reward == 0

        # Moving down should NOT move left when there are obstacles in either the cell below or the cell where it would move due to the wind
        env.setState(2)
        next_state, reward, _, _ = env.step(Direction2D.DOWN.value)
        assert next_state == 2  # There is an obstacle in states 5 and 6, and 6 is below 2
        assert env.getState() == next_state
        assert reward == 0

        env.setState(1)
        next_state, reward, _, _ = env.step(Direction2D.DOWN.value)
        assert next_state == 1  # There is an obstacle in state 5, and although the wind would tell the agent to end up in state 4, it can't because there is an obstacle just below the current state at 1.
        assert env.getState() == next_state
        assert reward == 0

        # Moving left should move just one cell to the left when the wind ALSO goes to the left, as opposed to moving *two* cells to the left
        env.setState(2)
        next_state, reward, _, _ = env.step(Direction2D.LEFT.value)
        assert next_state == 1
        assert env.getState() == next_state
        assert reward == 0


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Create the test suites
    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_Support_EnvGridworld2D_WithObstacles("test_show_gridworld"))
    test_suite.addTest(Test_Support_EnvGridworld2D_WithObstacles("test_transition_probabilities"))
    test_suite.addTest(Test_Support_EnvGridworld2D_WithObstacles("test_method_get_adjacent_states"))
    test_suite.addTest(Test_Support_EnvGridworld2D_WithObstacles("test_method_get_opposite_direction"))

    test_suite.addTest(Test_Support_EnvGridworld2D_WithObstaclesAndWind("test_move"))

    # Run the test suites
    runner.run(test_suite)
