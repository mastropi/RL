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
from timeit import default_timer as timer
import copy

import numpy as np

from Python.lib.simulators import SetOfStates, discrete
from Python.lib.environments.gridworlds import EnvGridworld1D
from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.episodic.discrete import td
from Python.lib.agents.policies import probabilistic


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


class Test_Class_Simulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Test environment
        nS = 5
        cls.env1d = EnvGridworld1D(length=nS, rewards_dict={nS-1: +1.0}, reward_default=0.0)
        print(f"Policy in {nS}-state gridworld:")
        for k, v in cls.env1d.P.items():
            print(f"State {k}: {v}")
        print(f"Terminal states in {nS}-state gridworld: {cls.env1d.getTerminalStates()}")
        print(f"Terminal rewards: {cls.env1d.getRewards()}")

        # Policy
        policy_probabilities = [0.1, 0.9]   #[0.5, 0.5]) #[0.9, 0.1])
        cls.policy = probabilistic.PolGenericDiscrete(cls.env1d, dict({0: [0.0, 1.0], nS-1: [0.0, 1.0]}), policy_default=policy_probabilities)

    def test_run_single_and_run_single_continuing_task_give_same_results_for_continuing_learning_task(self):
        """ (2024/02/08)
        Tests that the two methods defined in the discrete.Simulator class to run the simulation give the same results for a CONTINUING learning task

        This test was created to make sure that results don't change as I refactor the _run_single_continuing_task() method into a simpler implementation,
        and possibly into a fairly different implementation where dealing with continuing learning tasks derived from *naturally* episodic learning tasks
        (as those associated to environments containing terminal states) is a VERY particular case of the method's execution.

        I see the following advantages of using _run_single_continuing_task() (with its new-yet-to-come implementation) to run simulations for continuing learning tasks:
        - Its call is easier to understand (e.g. we do NOT need to think about parameters `nepisodes` and `max_time_steps_per_episode`, as there are no episodes).
        - It is easier to perform the iterative update of the average reward (e.g. there is no need to wait for the end of the episode to update it).
        - The upper class of the learner can be GenericLearner as opposed to Learner (defined in discrete/__init__.py) as the former does NOT deal with episodes.
        - It should be easier to extend the learning process to TD(lambda) learning, with lambda > 0 (currently the learning process is deemed to work ONLY for TD(0))
        and this includes Monte-Carlo learning.
        - ... possibly other advantages ...
        """
        print("\n*** Running test " + self.id() + " ***")

        # Learner parameters
        alpha = 1.0
        gamma = 1.0
        alpha_min = 0.0
        adjust_alpha = True

        # Simulation parameters
        # Note that we test different values for the simulation time steps so that we can test different ending points of the simulation,
        # e.g. at a terminal state (case max_time_steps = 50 for seed=1713, nepisodes = 10), at a start state (case max_time_steps = 51), at any other state (most of other cases)
        nepisodes = 10
        all_max_time_steps_to_test = [1, 2, 3, 4, 5,
                                      45, 46, 47, 48, 49, 50, 51, 52]
        seed = 1713

        # Expected values for selected max_time_steps cases (because it is cumbersome to copy all expected values for all tested cases!)
        # 'C' stands for state counts.
        expected_values = dict({45: {'V': [-0.12097312,  0.10998136,  0.46934819,  0.67405153, -0.25618422],
                                     'Q': [[0.        , -0.12097312, -0.12797619,  0.16286081, -0.19404762,
                                            0.55227267,  0.        ,  0.67405153, -0.25618422, -0.25618422]],
                                     'C': [10, 11, 9, 8, 8]
                                     },
                                # DM-2025/01/14: This is one case where the simulation ends at a start state (i.e. just after transitioning from a terminal state to a start state)
                                # So, it's a VERY special situation in terms of the learning process as currently implemented because it is made up of PATCHES,
                                # until we more cleanly implement the computation of the average reward in the CONTINUING task
                                # (which is now computed as an adjustment of the EPISODIC average reward --see Learner.update_average_reward() for more details
                                # and the implementations of _run_single(), _run_single_continuing_task() and run_exploration_and_learn_value_functions() in discrete.Simulator
                                # for implementation details of the PATCHES --look for the string "TEMPORARY")
                                # NOTE that the state counts list ('C') is the same as the above case `45`, even if there is "one more step" in this case.
                                # The reason is that the last step of the current case is from a terminal to a start state and we don't count that step as a valid step
                                # because o.w. the sum of the state counts would not coincide with the number of simulation steps, as it should (tested below).
                                46: {'V': [-0.12097312,  0.10998136,  0.46934819,  0.67405153, -0.26102196],
                                     'Q': [[0.        , -0.12097312, -0.12797619,  0.16286081, -0.19404762,
                                            0.55227267,  0.        ,  0.67405153, -0.26102196, -0.26102196]],
                                     'C': [10, 11, 9, 8, 8]
                                     },
                                49: {'V': [-0.11578753,  0.12543584,  0.47242722,  0.67405153, -0.26102196],
                                     'Q': [[0.        , -0.11578753, -0.12797619,  0.17611824, -0.19404762,
                                            0.54647998,  0.        ,  0.67405153, -0.26102196, -0.26102196]],
                                     'C': [11, 12, 10, 9, 8]
                                     },
                                50: {'V': [-0.11578753,  0.12543584,  0.47242722,  0.66194192, -0.26102196],
                                     'Q': [[0.        , -0.11578753, -0.12797619,  0.17611824, -0.19404762,
                                            0.54647998,  0.        ,  0.66194192, -0.26102196, -0.26102196]],
                                     'C': [11, 12, 10, 9, 9]
                                     },
                                # This is another case where the simulation ends at a start state (i.e. just after transitioning from a terminal state to a start state)
                                # See more details about these cases in the comment above for entry `46`, where the same thing happens.
                                51: {'V': [-0.11578753,  0.12543584,  0.47242722,  0.66194192, -0.26449264],
                                     'Q': [[0.        , -0.11578753, -0.12797619,  0.17611824, -0.19404762,
                                            0.54647998,  0.        ,  0.66194192, -0.26449264, -0.26449264]],
                                     'C': [11, 12, 10, 9, 9]
                                     },
                                52: {'V': [-0.11039147,  0.12543584,  0.47242722,  0.66194192, -0.26449264],
                                     'Q': [[0.        , -0.11039147, -0.12797619,  0.17611824, -0.19404762,
                                            0.54647998,  0.        ,  0.66194192, -0.26449264, -0.26449264]],
                                     'C': [12, 13, 10, 9, 9]
                                     }
                                })
        expected_average_reward_values = dict({1: 0, 2: 0, 3: 0,
                                               45: 0.1739130,
                                               46: 0.1739130,
                                               49: 0.16,
                                               50: 0.1764706,
                                               51: 0.1764706,
                                               52: 0.1698113})

        for case, max_time_steps in enumerate(all_max_time_steps_to_test):
            print(f"\n*** Case {case+1} of {len(all_max_time_steps_to_test)}: Testing simulators _run_single() and _run_single_continuing_task() under the CONTINUING learning task in a 1D gridworld for max_time_steps={max_time_steps}...")
            # Learner and agent definition
            learner_td = td.LeaTDLambda(self.env1d,
                                        criterion=LearningCriterion.AVERAGE,
                                        task=LearningTask.CONTINUING,
                                        alpha=alpha, gamma=gamma, lmbda=0.0,
                                        adjust_alpha=adjust_alpha, adjust_alpha_by_episode=False,
                                        alpha_min=alpha_min,
                                        debug=False)
            agent_td = GenericAgent(self.policy, learner_td)
            sim = discrete.Simulator(self.env1d, agent_td, debug=False)

            # Simulation using _run_single()
            print(f"Running simulation with _run_single()...")
            time_start = timer()
            state_values_td_1, action_values_td_1, advantage_values_td_1, state_counts_td_1, _, _, learning_info_1 = \
                sim._run_single(nepisodes=nepisodes, max_time_steps=max_time_steps, max_time_steps_per_episode=+np.Inf,
                                seed=seed, verbose=True, verbose_period=10)
            time_td_1 = timer() - time_start
            avg_reward_td_1 = learner_td.getAverageReward()
            visited_states_1 = learner_td.getStates()
            recorded_rewards_1 = learner_td.getRewards()
            avg_reward_raw_td_1 = np.mean(learner_td.getRewards())
            print(f"State counts: {state_counts_td_1}")
            print(f"Average reward (as stored in learner): {avg_reward_td_1}")
            print(f"Average reward (raw, i.e. plain average of observed rewards): {avg_reward_raw_td_1}")
            print("_run_single() took {:.4f} seconds".format(time_td_1))
            assert np.sum(state_counts_td_1) == len(learner_td.getStates()), f"The sum of the state counts ({np.sum(state_counts_td_1)}) must coincide with the trajectory length ({len(learner_td.getStates())})"
            assert np.isclose(avg_reward_td_1, avg_reward_raw_td_1), "[TD1] The average reward stored in the learner must be the same as the raw average reward computed as plain average of the observed rewards"

            # Simulation using _run_single_continuing_task()
            print(f"\nRunning simulation with _run_single_continuing_task()...")
            time_start = timer()
            state_values_td_2, action_values_td_2, advantage_values_td_2, state_counts_td_2, _, _, learning_info_2 = \
                sim._run_single_continuing_task(max_time_steps=max_time_steps, max_time_steps_per_episode=+np.Inf,
                                                seed=seed, verbose=True, verbose_period=10)
            time_td_2 = timer() - time_start
            avg_reward_td_2 = learner_td.getAverageReward()
            visited_states_2 = learner_td.getStates()
            recorded_rewards_2 = learner_td.getRewards()
            avg_reward_raw_td_2 = np.mean(learner_td.getRewards())
            print(f"State counts: {state_counts_td_2}")
            print(f"Average reward (as stored in learner): {avg_reward_td_2}")
            print(f"Average reward (raw, i.e. plain average of observed rewards): {avg_reward_raw_td_2}")
            print("_run_single_continuing_task() took {:.4f} seconds".format(time_td_2))
            assert np.sum(state_counts_td_2) == len(learner_td.getStates()), f"The sum of the state counts ({np.sum(state_counts_td_2)}) must coincide with the trajectory length ({len(learner_td.getStates())})"
            assert np.isclose(avg_reward_td_2, avg_reward_raw_td_2), "[TD2] The average reward stored in the learner must be the same as the raw average reward computed as plain average of the observed rewards"

            # Simulation using run_exploration_and_learn_value_functions()
            print(f"\nRunning simulation with run_exploration_and_learn_value_functions()...")
            time_start = timer()
            learner_exploration, nsteps = sim.run_exploration_and_learn_value_functions(max_time_steps=max_time_steps, seed=seed, verbose=True, verbose_period=10)
            time_td_3 = timer() - time_start
            state_counts_td_3 = learner_exploration.getStateCounts()
            learning_info_3 = {'nsteps': nsteps}
            state_values_td_3 = learner_exploration.getV().getValues()
            action_values_td_3 = learner_exploration.getQ().getValues()
            avg_reward_td_3 = learner_exploration.getAverageReward()
            visited_states_3 = learner_exploration.getStates()
            recorded_rewards_3 = learner_exploration.getRewards()
            avg_reward_raw_td_3 = np.mean(learner_exploration.getRewards())
            print(f"State counts: {state_counts_td_3}")
            print(f"Average reward (as stored in learner): {avg_reward_td_3}")
            print(f"Average reward (raw, i.e. plain average of observed rewards): {avg_reward_raw_td_3}")
            print("run_exploration_and_learn_value_functions() took {:.4f} seconds".format(time_td_3))
            assert np.sum(state_counts_td_3) == len(learner_exploration.getStates()), f"The sum of the state counts ({np.sum(state_counts_td_3)}) must coincide with the trajectory length ({len(learner_exploration.getStates())})"
            assert np.isclose(avg_reward_td_3, avg_reward_raw_td_3), "[TD3] The average reward stored in the learner must be the same as the raw average reward computed as plain average of the observed rewards"

            assert nepisodes == 10 and seed == 1713
            assert learning_info_1['nsteps'] == learning_info_2['nsteps']
            assert all(state_counts_td_1 == state_counts_td_2)
            assert np.allclose(state_values_td_1, state_values_td_2)
            assert np.allclose(action_values_td_1, action_values_td_2)
            assert np.isclose(avg_reward_td_1, avg_reward_td_2)
            assert visited_states_1 == visited_states_2
            assert recorded_rewards_1 == recorded_rewards_2

            assert learning_info_1['nsteps'] == learning_info_3['nsteps']
            assert all(state_counts_td_1 == state_counts_td_3)
            assert np.allclose(state_values_td_1, state_values_td_3)
            assert np.allclose(action_values_td_1, action_values_td_3)
            assert np.isclose(avg_reward_td_1, avg_reward_td_3)
            assert visited_states_1 == visited_states_3
            assert recorded_rewards_1 == recorded_rewards_3

            # Check whether the estimated values are the expected ones (in order to find out whether something equally affected the outcome of both tested methods above) or,
            # if there is a difference between the two outcomes above, which outcome is correct.
            assert expected_average_reward_values.get(max_time_steps, None) is None or np.isclose(avg_reward_td_1, expected_average_reward_values.get(max_time_steps))
            assert expected_values.get(max_time_steps, None) is None or \
                    np.allclose(state_values_td_1, expected_values.get(max_time_steps)['V']) and \
                    np.allclose(action_values_td_1, np.array(expected_values.get(max_time_steps)['Q']).reshape(-1)) and \
                    all(state_counts_td_1 == expected_values.get(max_time_steps)['C'])


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_Class_SetOfStates("test_basic_functionality"))
    test_suite.addTest(Test_Class_Simulator("test_run_single_and_run_single_continuing_task_give_same_results_for_continuing_learning_task"))
    runner.run(test_suite)
