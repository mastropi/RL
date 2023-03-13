# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:55:27 2022

@author: Daniel Mastropietro
@description: Unit tests for functions and methods defined in simulators.queues.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest

import numpy as np

from Python.lib.simulators import SetOfStates


class Test_Class_SetOfStates(unittest.TestCase):

    def test_basic_functionality(self):
        # Using set_boundaries attribute
        print("Testing the SetOfStates class...")
        print("0) Test Getters multidimensional")
        set_states = SetOfStates(states={(1, 1, 0), (0, 1, 2), (1, 2, 2), (2, 2, 0)})
        set_boundaries = SetOfStates(set_boundaries=(3, 5, 4))
        # getDimension()
        assert set_states.getStateDimension() == 3
        assert set_boundaries.getStateDimension() == 3
        # getStates()
        assert set_states.getStates() == {(1, 1, 0), (0, 1, 2), (1, 2, 2), (2, 2, 0)}
        assert set_boundaries.getSetBoundaries() == (3, 5, 4)
        # getStorageFormat()
        assert set_states.getStorageFormat() == "states"
        assert set_boundaries.getStorageFormat() == "set_boundaries"

        print("0) Test Getters unidimensional")
        set_states = SetOfStates(states={3, 5})
        set_boundaries = SetOfStates(set_boundaries=5)
        # getStateDimension()
        assert set_states.getStateDimension() == 1
        assert set_boundaries.getStateDimension() == 1
        # getStates()
        assert set_states.getStates() == {3, 5}
        assert set_boundaries.getSetBoundaries() == 5
        # getStorageFormat()
        assert set_states.getStorageFormat() == "states"
        assert set_boundaries.getStorageFormat() == "set_boundaries"

        print("1) MULTIDIMENSIONAL STATES")
        print("A) Using set_boundaries in constructor on multi-dimensional states:")
        seed = 1313
        np.random.seed(seed)
        set_boundaries = SetOfStates(set_boundaries=(3, 5, 4))
        selected_states = set_boundaries.random_choice(4)
        expected_states = [(3, 3, 4), (3, 1, 0), (1, 4, 4), (2, 5, 0)]
        print("Selected states: ", selected_states)
        assert seed == 1313 and selected_states == expected_states
        # Check that the range of values of each dimension in the selected states always fall between 0 and the corresponding value given in set_boundaries passed to the SetOfStates constructor
        print("Checking range of the selected states in each dimension (of each of 6 randomly selected states) over 20 replications...")
        for _ in range(20):
            selected_states = set_boundaries.random_choice(6)
            for state in selected_states:
                for s in range(len(state)):
                    assert 0 <= state[s] <= set_boundaries.getSetBoundaries()[s]
        # Check the explicit enumeration of states in the set
        set_boundaries = SetOfStates(set_boundaries=(1, 2, 0))
        all_states = set_boundaries.getStates()
        print(f"Case 1: Boundaries = (1, 2, 0) --> Set of ALL states ({len(all_states)}): ", all_states)
        assert set_boundaries.getStates() == { (0, 0, 0),
                                               (0, 1, 0), (1, 1, 0),
                                               (0, 2, 0), (1, 2, 0),
                                               (1, 0, 0), (1, 1, 0), (1, 2, 0)}
        set_boundaries = SetOfStates(set_boundaries=(2, 3, 1))
        all_states = set_boundaries.getStates()
        all_states_restricted_exactly_one = set_boundaries.getStates(exactly_one_dimension_at_boundary=True)
        print(f"Case 2: Boundaries = (2, 3, 1) --> Set of ALL states ({len(all_states)}): ", all_states)
        print(f"Case 2: Boundaries = (2, 3, 1) --> Set of ALL states with ONLY ONE dimension at boundary ({len(all_states_restricted_exactly_one)}): ", all_states_restricted_exactly_one)
        assert all_states_restricted_exactly_one == all_states.difference({ (2, 3, 0), (2, 0, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1), (0, 3, 1), (1, 3, 1),
                                                                            (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0)})
        all_states_restricted_atleast_one = set_boundaries.getStates(at_least_one_dimension_at_boundary=True)
        print(f"Case 2: Boundaries = (2, 3, 1) --> Set of ALL states with AT LEAST ONE dimension at boundary ({len(all_states_restricted_atleast_one)}): ", all_states_restricted_atleast_one)
        assert all_states_restricted_atleast_one == all_states.difference({(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0)})

        # Using states attributes
        print("B) Using states in constructor on multi-dimensional states:")
        seed = 1313
        np.random.seed(seed)
        set_states = SetOfStates(states={(1, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, 2), (1, 2, 2)})
        selected_states = set_states.random_choice(4)
        expected_states = [(1, 1, 0), (1, 2, 2), (1, 1, 0), (0, 1, 2)]     # Note that states are repeated, i.e. the sampling is with replacement (OK!)
        print("Selected states: ", selected_states)
        assert seed == 1313 and selected_states == expected_states

        print("\n2) UNIDIMENSIONAL STATES")
        print("A) Using set_boundaries in constructor on unidimensional states:")
        set_boundaries = SetOfStates(set_boundaries=3)
        selected_states = set_boundaries.random_choice(4)
        expected_states = [3, 3, 3, 3]
        print("Selected states: ", selected_states)
        print("Set of ALL states: ", set_boundaries.getStates())
        assert set_boundaries.getStates() == {0, 1, 2, 3}
        assert selected_states == expected_states

        # Using states attributes
        print("B) Using states in constructor on unidimensional states (single state in set):")
        set_states = SetOfStates(states={3})
        selected_states = set_states.random_choice(4)
        expected_states = [3, 3, 3, 3]
        print("Selected states: ", selected_states)
        assert selected_states == expected_states

        print("C) Using states in constructor on unidimensional states (multiple states in set):")
        seed = 1313
        np.random.seed(seed)
        set_states = SetOfStates(states={3, 2, 5})
        selected_states = set_states.random_choice(4)
        expected_states = [2, 2, 5, 3]
        print("Selected states: ", selected_states)
        assert seed == 1313 and selected_states == expected_states

        print("D) Check use of parameter `size` in random_choice() which can be either None or an integer")
        seed = 1313
        np.random.seed(seed)
        set_boundaries = SetOfStates(set_boundaries=(3, 5, 4))
        selected_states = set_boundaries.random_choice()
        assert selected_states == (3, 3, 0)     # A single element is returned
        np.random.seed(seed)
        selected_states = set_boundaries.random_choice(1)
        assert selected_states == [(3, 3, 0)]    # A list with one element is returned

        seed = 1313
        np.random.seed(seed)
        set_states = SetOfStates(states={(1, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, 2), (1, 2, 2)})
        selected_states = set_states.random_choice()
        assert selected_states == (1, 1, 0)     # A single element is returned
        np.random.seed(seed)
        selected_states = set_states.random_choice(1)
        assert selected_states == [(1, 1, 0)]    # A list with one element is returned

        print("\nOK")


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_Class_SetOfStates("test_basic_functionality"))
    runner.run(test_suite)
