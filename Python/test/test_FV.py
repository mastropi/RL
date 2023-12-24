# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 18:59:23 2023

@author: Daniel Mastropietro
@description: Unit tests for Fleming-Viot simulation and estimation.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest

import copy
import numpy as np
import pandas as pd

import Python.lib.environments.gridworlds as gridworlds
from Python.lib.estimators.fv import merge_proba_survival_and_phi, estimate_proba_stationary
from Python.lib.agents.learners.episodic.discrete.fv import LeaFV


class Test_Class_LeaFV_discretetime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gridworlds.EnvGridworld1D_OneTerminalState(length=5, rewards_dict={4: +1}, reward_default=0.0)
        absorption_set = {1}
        cls.absorption_boundary = max(absorption_set)
        activation_set = {cls.absorption_boundary + 1}
        cls.active_set = set(np.arange(cls.absorption_boundary + 1, cls.env.getNumStates()))
        cls.learner_fv = LeaFV(cls.env, N=5, T=10, absorption_set=absorption_set, activation_set=activation_set, TIME_RESOLUTION=1)

    def test_iterative_computation_of_FV_integral(self):
        "Tests the iterative calculation of the integral appearing in the FV estimator"
        print("\n*** Running test " + self.id() + " ***")
        # Create the particles and mock where they are so that we can have a Phi estimate
        envs = [copy.deepcopy(self.env) for _ in range(self.learner_fv.N)]
        # Mock the observed absorption times
        absorption_times = [3, 8, 13]
        assert sorted(absorption_times) == absorption_times, "The absorption times must be sorted increasingly"

        # Maximum time steps for which the evolution of the FV particles is mocked
        t_max = 20
        # Simulation seeds on which assertions are run
        seeds = [13, 17, 23, 1717]
        seed_for_adhoc_test = 1717

        # Expected results for each seed considered
        dict_expected = dict({  13: dict({'dict_phi': pd.DataFrame({ 't': [0, 2, 7, 8, 9, 13, 15, 17, 19, 20],
                                                                    'Phi': [0.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.0, 0.2, 0.0, 0.2]}),
                                          'integral': 0.6
                                          }),
                                17: dict({'dict_phi': pd.DataFrame({ 't': [0, 2, 7, 11, 13, 14, 16, 18],
                                                                    'Phi': [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.4, 0.2]}),
                                          'integral': 0.1333333
                                          }),
                                23: dict({'dict_phi': pd.DataFrame({ 't': [0, 2, 3, 4, 5, 8, 9, 10, 17, 18],
                                                                    'Phi': [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.4]}),
                                          'integral': 0.6
                                          }),
                                1717: dict({'dict_phi': pd.DataFrame({ 't': [0, 2, 7, 8, 9, 12, 14, 17, 19, 20],
                                                                    'Phi': [0.0, 0.0, 0.2, 0.0, 0.2, 0.4, 0.2, 0.0, 0.2, 0.4]}),
                                          'integral': 0.4666667
                                          }),
                              })

        for idx_seed, seed in enumerate(seeds):
            print(f"\n--- Running test for seed = {seed} ({idx_seed+1} of {len(seeds)}) ---")
            np.random.seed(seed)
            self.learner_fv.reset()

            # Initialize the particle's position
            for p in range(self.learner_fv.N):
                envs[p].setState(np.random.choice(list(self.active_set)))
            print(f"Initial state of particles: {[e.getState() for e in envs]}")

            # Mock the evolution of the FV particles
            # (To make implementation easier, state transitions can happen without regard of the transition probabilities defined in the environment)
            t = 0
            done = False
            while not done:
                # Mock the movement of the particles
                for p in range(self.learner_fv.N):
                    t += 1

                    # Needed for the test of the particular situation where Phi(t) changes at the same time when an absorption event happens
                    if seed == seed_for_adhoc_test and t == absorption_times[-2]:
                        # To understand this update of p, see comment below under the same IF condition
                        p = p - 1

                    # Retrieve the state before the system update
                    state_prev = envs[p].getState()
                    if t in absorption_times:
                        # Since t is an absorption time, we set the state to be at the absorption set boundary
                        state = self.absorption_boundary
                    # Test the particular situation where Phi(t) changes at the same time when an absorption event happens
                    elif seed == seed_for_adhoc_test and t == absorption_times[-2] - 1:
                        # Set the state to a state of interest (and change p to p-1 at the very end of the loop),
                        # so that, at the next iteration, the same particle that is updated now will be updated
                        # and this will allow us to test the case when Phi is updated at the same time of an absorption event.
                        state = list(self.learner_fv.states_of_interest)[0]
                    else:
                        state = np.random.choice(list(self.active_set))
                    envs[p].setState(state)
                    print(f"t={t} --> Particle p={p} moved from s={state_prev} --> {state}")
                    if t in absorption_times:
                        print(f"ABSORPTION!")
                        if seed != seed_for_adhoc_test:
                            # Reactivate the particle, EXCEPT for the case where we want to test the situation where Phi(t) changes at an absorption time
                            # (because if we reactivate the particle, it may end up at the state of interest and this would make Phi(t) not change,
                            # as the particle was coming from the state of interest).
                            state = np.random.choice(list(self.active_set))
                            envs[p].setState(state)
                            print(f"--> Particle p={p} reactivated to state s={state}")
                        print("----------------------")
                        print(f"Processing absorption time t_abs={t}...")
                        self.learner_fv._update_absorption_times(t)
                        self.learner_fv._update_phi_contribution(t)
                        self.learner_fv._update_integral()
                    self.learner_fv._update_phi(t, state_prev, state)

                    if t == t_max:
                        done = True

            # ASSERTIONS
            # Compute the after the fact survival probability (i.e. after all absorption times are observed)
            # and check the correctness of the iteratively computed integral of P(T>t)*Phi(t,x) for each state of interest x
            df_surv = pd.DataFrame({'t': [0] + absorption_times,
                                    'P(T>t)': [ n /len(absorption_times) for n in range(len(absorption_times), -1, -1)]})

            print(f"\n--- Seed {seed} ({idx_seed + 1} of {len(seeds)}):")

            # Assert the value of the integral
            for x in self.learner_fv.states_of_interest:
                df_proba_surv = merge_proba_survival_and_phi(df_surv, self.learner_fv.dict_phi[x])
                proba, integral = estimate_proba_stationary(df_proba_surv, 1.0)
                #print(f"Components of the integral:\n{df_proba_surv}")
                #print(f"Integral computed after the fact: {integral}")

                print(f"Observed integral[x={x}]: {self.learner_fv.dict_integral[x]}")
                assert np.isclose(self.learner_fv.dict_integral[x], integral), "The iterative method and the after the fact method must give the same value for the FV integral"

            # Assert the value of Phi(t,x) for each seed
            for x in self.learner_fv.states_of_interest:
                print(f"Observed Phi[x={x}]:\n{self.learner_fv.dict_phi[x]}")
                assert np.allclose(self.learner_fv.dict_phi[x], dict_expected[seed]['dict_phi'])

if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    test = True

    if test:
        runner = unittest.TextTestRunner()

        # unittest.getTestCaseNames()

        # Run all tests
        #unittest.main()

        # Create the test suites
        test_suite = unittest.TestSuite()
        test_suite.addTest(Test_Class_LeaFV_discretetime("test_iterative_computation_of_FV_integral"))

        # Run the test suite
        runner.run(test_suite)
    else:
        # Ad-hoc test for the FV estimator (in discrete time), which inspired a few unit tests run by this script

        # These are the modules that are needed for the execution below which I comment out because they are included above
        # but leave here in order to know what is really needed for this execution.
        #import copy
        #import Python.lib.environments.gridworlds as gridworlds
        #from Python.lib.estimators.fv import merge_proba_survival_and_phi, estimate_proba_stationary

        env = gridworlds.EnvGridworld1D_OneTerminalState(length=5, rewards_dict={4: +1}, reward_default=0.0)
        absorption_set = {1}
        absorption_boundary = max(absorption_set)
        activation_set = {absorption_boundary + 1}
        active_set = set(np.arange(absorption_boundary + 1, env.getNumStates()))
        learner_fv = LeaFV(env, N=5, T=10, absorption_set=absorption_set, activation_set=activation_set)
        # Create the particles and mock where they are so that we can have a Phi estimate
        envs = [copy.deepcopy(env) for _ in range(learner_fv.N)]
        # Mock the observed absorption times
        absorption_times = [3, 8, 13]

        t_max = 20      # Maximum time steps for which the evolution of the FV particles is mocked
        seeds = [13, 17, 23, 1313, 1713, 1717]
        for idx_seed, seed in enumerate(seeds):
            print(f"\n******** Running test for seed = {seed} ({idx_seed+1} of {len(seeds)})")
            np.random.seed(seed)
            learner_fv.reset()

            # Initialize the particle's position
            for p in range(learner_fv.N):
                envs[p].setState(np.random.choice(list(active_set)))
            print(f"Initial state of particles: {[e.getState() for e in envs]}")

            # Mock the evolution of the FV particles
            t = 0
            done = False
            while not done:
                # Mock the movement of the particles
                for p in range(learner_fv.N):
                    t += 1

                    # For special test of the special coincidence where Phi is to be updated at the same time an absorption event happens
                    if seed == 1717 and t == absorption_times[-2]:
                        # To understand this updated of p, see comment below under the same IF condition
                        p = p - 1
                        print(f"t={t}: ENTRO p decreased to p={p}")

                    state_prev = envs[p].getState()
                    if t in absorption_times:
                        # Since t is an absorption time, we set the state to be at the absorption set boundary
                        state = absorption_boundary
                    # For special test of the special coincidence where Phi is to be updated at the same time an absorption event happens
                    elif seed == 1717 and t == absorption_times[-2] - 1:
                        # Set the state to a state of interest (and change p to p-1 at the very end of the loop),
                        # so that, at the next iteration, the same particle that is updated now will be updated
                        # and this will allow us to test the case when Phi is updated at the same time of an absorption event.
                        state = list(learner_fv.states_of_interest)[0]
                        print(f"t={t}: ENTRO state forced to s={state}")
                    else:
                        state = np.random.choice(list(active_set))
                    envs[p].setState(state)
                    print(f"t={t} --> Particle p={p} moved from s={state_prev} --> {state}")
                    if t in absorption_times:
                        # Choose the new state of the absorbed particle as a random value among the active states (reactivation)
                        if seed != 1717:
                            state = np.random.choice(list(active_set))
                            envs[p].setState(state)
                            print(f"--> ABSORPTION! Particle p={p} reactivated to state s={state}")
                        print("----------------------")
                        print(f"Processing absorption time t_abs={t}...")
                        learner_fv._update_absorption_times(t)
                        learner_fv._update_phi_contribution(t)
                        learner_fv._update_integral()
                    learner_fv._update_phi(t, state_prev, state)
                    print("After UPDATE:")
                    print(f"Phi:\n{learner_fv.dict_phi}")
                    print(f"Indices for Phi:\n{learner_fv.dict_last_indices_phi_prior_to_absorption_times}")
                    print(f"CUMSUM Phi:\n{learner_fv.dict_phi_sum}")
                    print(f"Integral:\n{learner_fv.dict_integral}")
                    print()

                    if t == t_max:
                        done = True
            print(f"\nDONE!\n=======\nPhi:\n{learner_fv.dict_phi}")

            ## OK!
            # NEXT: (2023/12/21) Assert the following output at the end (for seed = 13):
            # ----------------------
            # Processing absorption time t_abs=13...
            # Phi:
            # {4:    Phi   t
            # 0  0.0   0
            # 0  0.0   2
            # 0  0.2   5
            # 0  0.4   6
            # 0  0.6   9
            # 0  0.8  12}
            # Indices for Phi:
            # {4: [0, 1, 3, 5]}
            # CUMSUM Phi:
            # {4:    Phi   t
            # 0  0.0   0
            # 0  0.0   3
            # 0  1.0   8
            # 0  3.0  13}
            # Integral:
            # {4: 3.5}

            # NEXT: (2023/12/21) Check that the integral gives the same as the integral computed after the fact, as opposed to iteratively as done above.
            print("\n========\nCheck with computation of the integral after the fact...")
            df_surv = pd.DataFrame({'t': [0] + absorption_times,
                                    'P(T>t)': [ n /len(absorption_times) for n in range(len(absorption_times), -1, -1)]})

            for x in learner_fv.states_of_interest:
                df_proba_surv = merge_proba_survival_and_phi(df_surv, learner_fv.dict_phi[x])
                proba, integral = estimate_proba_stationary(df_proba_surv, 1.0)
                print(f"Components of the integral:\n{df_proba_surv}")
                print(f"Integral computed after the fact: {integral}")

                print(f"\nAsserting equality of integrals (iterative vs. after the fact) for seed = {seed} ({idx_seed+1} of {len(seeds)})")
                assert np.isclose(integral, learner_fv.dict_integral[x]), "The iterative method and the after the fact method must give the same value for the FV integral"
                print("OK!")
