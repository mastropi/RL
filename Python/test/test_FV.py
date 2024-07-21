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
from Python.lib.agents.learners import LearningCriterion
from Python.lib.agents.learners.episodic.discrete.fv import LeaFV


class Test_Class_LeaFV_discretetime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = gridworlds.EnvGridworld1D(length=5, rewards_dict={4: +1}, reward_default=0.0)

    def test_iterative_computation_of_FV_integral(self):
        "Tests the iterative calculation of the integral appearing in the FV estimator when survival times are observed in increasing order"
        print("\n*** Running test " + self.id() + " ***")

        # Define the FV learner
        N = 5
        absorption_set = set(range(2))
        absorption_boundary = max(absorption_set)
        activation_set = {absorption_boundary + 1}
        learner_fv = LeaFV(self.env, N=N, T=10, absorption_set=absorption_set, activation_set=activation_set, states_of_interest=self.env.getTerminalStates()) #, TIME_RESOLUTION=1) # This was commented out because the default value of TIME_RESOLUTION is supposed to be 1 and by omitting its value here we want to stress this out, as normally when creating the LeaFV object, we will forget about setting this parameter... This value of TIME_RESOLUTION=1 should we set for discrete-time Markov chains.

        # Create the particles and mock where they are so that we can have a Phi estimate
        envs = [copy.deepcopy(self.env) for _ in range(learner_fv.N)]
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
            learner_fv.reset()

            # Initialize the particle's position
            for p in range(learner_fv.N):
                envs[p].setState(np.random.choice(list(learner_fv.active_set)))
            print(f"Initial state of particles: {[e.getState() for e in envs]}")

            # Mock the evolution of the FV particles
            # (To make implementation easier, state transitions can happen without regard of the transition probabilities defined in the environment)
            t = 0
            done = False
            while not done:
                # Mock the movement of the particles
                for p in range(learner_fv.N):
                    t += 1

                    # Needed for the test of the particular situation where Phi(t) changes at the same time when an absorption event happens
                    if seed == seed_for_adhoc_test and t == absorption_times[-2]:
                        # To understand this update of p, see comment below under the same IF condition
                        p = p - 1

                    # Retrieve the state before the system update
                    state_prev = envs[p].getState()
                    if t in absorption_times:
                        # Since t is an absorption time, we set the state to be at the absorption set boundary
                        state = absorption_boundary
                    # Test the particular situation where Phi(t) changes at the same time when an absorption event happens
                    elif seed == seed_for_adhoc_test and t == absorption_times[-2] - 1:
                        # Set the state to a state of interest (and change p to p-1 at the very end of the loop),
                        # so that, at the next iteration, the same particle that is updated now will be updated
                        # and this will allow us to test the case when Phi is updated at the same time of an absorption event.
                        state = list(learner_fv.states_of_interest)[0]
                    else:
                        state = np.random.choice(list(learner_fv.active_set))
                    envs[p].setState(state)
                    print(f"t={t} --> Particle p={p} moved from s={state_prev} --> {state}")
                    if t in absorption_times:
                        print(f"ABSORPTION!")
                        if seed != seed_for_adhoc_test:
                            # Reactivate the particle, EXCEPT for the case where we want to test the situation where Phi(t) changes at an absorption time
                            # (because if we reactivate the particle, it may end up at the state of interest and this would make Phi(t) not change,
                            # as the particle was coming from the state of interest).
                            state = np.random.choice(list(learner_fv.active_set))
                            envs[p].setState(state)
                            print(f"--> Particle p={p} reactivated to state s={state}")
                        print("----------------------")
                        print(f"Processing absorption time t_abs={t}...")
                        learner_fv.update_integral(t, fixed_sample_size=True)
                    learner_fv.update_phi(t, state_prev, state)

                    if t == t_max:
                        done = True

            # Kill the particles that have not been absorbed (censored particles) at t=t_max, so that we can consider the contribution
            n_particles_not_absorbed = N - len(absorption_times)

            # Update the integral taking into account this last absorption of all remaining particles, which makes the estimated survival probability at t_max be equal to 0.
            # NOTE that we only need to update the integral ONCE, even when more than one particle has yet to be absorbed.
            learner_fv.update_integral(t, fixed_sample_size=True)

            # ASSERTIONS
            # Compute the after the fact survival probability (i.e. after all absorption times are observed)
            # and check the correctness of the iteratively computed integral of P(T>t)*Phi(t,x) for each state of interest x
            _absorption_times_including_censoring = absorption_times + list(np.repeat(t_max, n_particles_not_absorbed))
            df_surv = pd.DataFrame({'t': [0] + _absorption_times_including_censoring,
                                    'P(T>t)': [ n /len(_absorption_times_including_censoring) for n in range(len(_absorption_times_including_censoring), -1, -1)]})

            print(f"\n--- Seed {seed} ({idx_seed + 1} of {len(seeds)}):")

            # Assert the value of Phi and of the integral, and check that the iteratively computed integral gives the same result as the integral computed in one shot at the end
            for x in learner_fv.states_of_interest:
                print(f"CHECK Phi[x={x}]")
                print(f"Observed Phi[x={x}]:\n{learner_fv.dict_phi[x]}")
                assert np.allclose(learner_fv.dict_phi[x], dict_expected[seed]['dict_phi'])
                print("(OK)")

                # Compute the integral
                print(f"\nCHECK Integral[x={x}]")
                df_proba_surv_and_phi = merge_proba_survival_and_phi(df_surv, learner_fv.dict_phi[x])
                _, integral = estimate_proba_stationary(df_proba_surv_and_phi, 1.0) # We pass a dummy expected reabsorption time value of 1.0 as we are not interested in checking the stationary probability value
                print(f"Components of the integral:\n{df_proba_surv_and_phi}")
                print(f"One-shot computed Integral[x={x}]: {integral}")

                print(f"Iteratively computed Integral[x={x}]: {learner_fv.dict_integral[x]}")
                assert np.isclose(learner_fv.dict_integral[x], integral), "The iterative method and the after the fact method must give the same value for the FV integral"
                print("(OK)")


    def test_computation_of_phi_as_function_of_start_state_and_action(self):
        "Tests the calculation of Phi(t,x) for different start states s using mocked particle trajectories"
        print("\n*** Running test " + self.id() + " ***")

        # Define the FV learner with very few particles (e.g. 3 particles)
        absorption_set = {0}
        absorption_boundary = max(absorption_set)
        activation_set = {absorption_boundary + 1}
        learner_fv = LeaFV(self.env, N=3, T=10, absorption_set=absorption_set, activation_set=activation_set, states_of_interest=self.env.getTerminalStates(), criterion=LearningCriterion.DISCOUNTED, gamma=0.9)

        # Now we mock the evolution of the particles in the FV system
        # We do so by creating a data frame index by each time the particle system is updated, i.e. indexed by the FV system's clock
        # The particle that is updated at each time is defined by a separate list containing that information
        df_trajectories = pd.DataFrame({0: [2, 1, 2, 2, 2, 2, 3, 4, 4, 4, 3, 2],
                                        1: [3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        2: [2, 2, 2, 2, 3, 4, 4, 4, 3, 4, 4, 4]})
        updated_particle = [0, 0, 1, 2, 2, 0, 0, 2, 2, 0, 0]
        assert len(updated_particle) == len(df_trajectories) - 1

        # Particle system evolution based on the df_trajectories data frame
        # Note that the FIRST row in each particle's trajectory corresponds to the first time at which the particle is moved
        # The initial state of each particle is thus the value of 'state' at the first row.
        # Particles, whose trajectories will evolve as per the df_trajectories data frame just created
        particles = [copy.deepcopy(learner_fv.env) for _ in range(learner_fv.getNumParticles())]
        # Store the start states at the first row (so that we know at which state each particle is at the onset of the experiment)
        start_states = [df_trajectories[p].iloc[0] for p in range(learner_fv.getNumParticles())]

        for p, _particle in enumerate(particles):
            _particle.setState(start_states[p])
            _particle.setStoreTrajectoryFlag(True)
            _particle.reset_trajectory()
            assert np.allclose(_particle.getTrajectory(), [0, start_states[p], -1, start_states[p], self.env.getReward(start_states[p])], equal_nan=True), \
                "At the onset of the experiment, each particle's trajectory must have only ONE record having its state and next_state equal to their start state, and action equal to -1 (i.e. no action)"
        for t, p in enumerate(updated_particle):
            print(f"\ntime = {t+1}: Particle #{p} MOVES")
            state = df_trajectories.iloc[t, p]
            action = int( 0.5*(df_trajectories.iloc[t+1, p] - df_trajectories.iloc[t, p]) + 0.5 )
            next_state = df_trajectories.iloc[t+1, p]
            reward = self.env.getReward(next_state)
            particles[p].update_trajectory(t+1, state, action, next_state, reward)  # We store t+1 because t starts at 0, which is the time at which the particle is at its initial state, and in the particle's trajectory we store the time at which the particle transitions to the NEXT state (i.e. the trajectory is continuous from the right)
            for pp, _particle in enumerate(particles):
                print(f"Particle #{pp}:\n{_particle.getTrajectory()}")

        # Update the value of Phi(t,x) for each state of interest x defined in the FV learner (which by default is the set of terminal states of the environment)
        learner_fv.update_phi_function(particles)
        for x in learner_fv.states_of_interest:
            print(f"\n*** Phi(t, x={x})***")
            for s in sorted(learner_fv.dict_phi_for_state.keys()):
                print(f"s = {s}:")
                print(learner_fv.dict_phi_for_state[s][x])
                for a in range(self.env.getNumActions()):
                    print(f"s = {s}, a = {a}:")
                    print(learner_fv.dict_phi_for_state_action[s][a][x])
                print("")

        # Check expected results
        # I verified that these results make sense and seem to be correct!
        expected_dict_phi_for_state = dict({1: {4: pd.DataFrame({'t': np.arange(6), 'Phi': [0, 0, 0, 1, 0, 0]}, columns=['t', 'Phi'])},
                                            2: {4: pd.DataFrame({'t': np.arange(7), 'Phi': [0, 0, 0.666667, 0, 0.666667, 0, 0]}, columns=['t', 'Phi'])},
                                            3: {4: pd.DataFrame({'t': np.arange(4), 'Phi': [0, 0.6, 0, 0.5]}, columns=['t', 'Phi'])},
                                            4: {4: pd.DataFrame({'t': np.arange(3), 'Phi': [1, 0, 0.5]}, columns=['t', 'Phi'])}
                                        })
        expected_dict_phi_for_state_action = dict({ 1: {0: {4: pd.DataFrame({'t': [0.0], 'Phi': [0.0]}, columns=['t', 'Phi'])},
                                                        1: {4: pd.DataFrame({'t': np.arange(6), 'Phi': [0, 0, 0, 1.0, 0, 0]}, columns=['t', 'Phi'])}
                                                       },
                                                    2: {0: {4: pd.DataFrame({'t': np.arange(7), 'Phi': [0, 0, 0, 0, 1.0, 0, 0]}, columns=['t', 'Phi'])},
                                                        1: {4: pd.DataFrame({'t': np.arange(5), 'Phi': [0, 0, 1.0, 0, 0.5]}, columns=['t', 'Phi'])}
                                                        },
                                                    3: {0: {4: pd.DataFrame({'t': np.arange(2), 'Phi': [0, 0]}, columns=['t', 'Phi'])},
                                                        1: {4: pd.DataFrame({'t': np.arange(4), 'Phi': [0, 1.0, 0, 0.5]}, columns=['t', 'Phi'])},
                                                        },
                                                    4: {0: {4: pd.DataFrame({'t': np.arange(3), 'Phi': [1, 0, 0.5]}, columns=['t', 'Phi'])},
                                                        1: {4: pd.DataFrame({'t': [0.0], 'Phi': [1.0]}, columns=['t', 'Phi'])},
                                                        }
                                        })
        assert learner_fv.dict_phi_for_state.keys() == learner_fv.active_set
        assert learner_fv.dict_phi_for_state.keys() == expected_dict_phi_for_state.keys()
        for s in learner_fv.active_set:
            assert learner_fv.dict_phi_for_state[s].keys() == learner_fv.states_of_interest
            for x in learner_fv.states_of_interest:
                # Note: For a reason I don't understand, I have to convert the data frame values of learner_fv.dict_phi_for_state[s][x] to float...
                # otherwise I get the error message "unfunc `isfinite()`..." or similar.
                assert np.allclose(learner_fv.dict_phi_for_state[s][x].astype(float), expected_dict_phi_for_state[s][x])

        assert learner_fv.dict_phi_for_state_action.keys() == learner_fv.active_set
        for s in learner_fv.active_set:
            assert learner_fv.dict_phi_for_state_action[s].keys() == set(np.arange(self.env.getNumActions()))
            for a in range(self.env.getNumActions()):
                assert learner_fv.dict_phi_for_state_action[s][a].keys() == learner_fv.states_of_interest
                for x in learner_fv.states_of_interest:
                    # Note: For a reason I don't understand, I have to convert the data frame values of learner_fv.dict_phi_for_state[s][x] to float...
                    # otherwise I get the error message "unfunc `isfinite()`..." or similar.
                    assert np.allclose(learner_fv.dict_phi_for_state_action[s][a][x].astype(float), expected_dict_phi_for_state_action[s][a][x])


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
        test_suite.addTest(Test_Class_LeaFV_discretetime("test_computation_of_phi_as_function_of_start_state_and_action"))

        # Run the test suite
        runner.run(test_suite)
    else:
        # Ad-hoc test for the FV estimator (in discrete time), which inspired a few unit tests run by this script

        #--- Test 1: (2023/12) FV learning of value functions under the DISCOUNTED learning criterion
        # These are the modules that are needed for the execution below which I may have commented out because they are included above
        # but leave here in order to know what is really needed for this execution.
        import numpy as np
        from Python.lib.environments import gridworlds
        from Python.lib.agents.learners import LearningCriterion
        from Python.lib.agents.policies import probabilistic
        from Python.lib.agents import GenericAgent
        from Python.lib.agents.learners.episodic.discrete.fv import LeaFV
        from Python.lib.simulators.discrete import Simulator as DiscreteSimulator

        # Note: the EnvGridworld1D environment always starts at 0, so no need to define an initial state distribution (isd)
        nS = 5
        env = gridworlds.EnvGridworld1D(length=nS, rewards_dict={nS-1: +1}, reward_default=0.0)
        absorption_set = set(range(2))              # ALL states in A should be part of the absorption set, because the set of active states in LeaFV is determined from the complement of the absorption set, and the active set is needed to define the dictionaries of the survival probability estimates.
        absorption_boundary = max(absorption_set)
        activation_set = {absorption_boundary + 1}
        active_set = set(np.arange(absorption_boundary + 1, env.getNumStates()))
        print(f"Absorption set: {absorption_set}")
        print(f"Activation set: {activation_set}")
        learner_fv = LeaFV(env, N=5, T=10, absorption_set=absorption_set, activation_set=activation_set, criterion=LearningCriterion.DISCOUNTED, gamma=0.9)
        policy = probabilistic.PolGenericDiscrete(env, dict({0: [0.0, 1.0], env.getNumStates()-1: [0.0, 1.0]}), policy_default=[0.5, 0.5]) #[0.9, 0.1])
        agent_fv = GenericAgent(policy, learner_fv)

        # Simulation
        seed = 1717
        nepisodes = 20
        sim = DiscreteSimulator(env, agent_fv, debug=False)
        state_values, action_values, advantage_values, state_counts, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv  = \
            sim.run(nepisodes=nepisodes,
                    max_time_steps_per_episode=np.Inf,
                    max_time_steps_fv=100,
                    seed=seed,
                    verbose=True, verbose_period=1,
                    plot=False)




        #--- Test 2: (2024/01/09) Phi(t,x) estimate as a function of the start state and state-action
        import copy
        import numpy as np
        import pandas as pd
        from Python.lib.environments import gridworlds
        from Python.lib.agents.learners import LearningCriterion
        from Python.lib.agents.learners.episodic.discrete.fv import LeaFV

        # Note: the EnvGridworld1D environment always starts at 0, so no need to define an initial state distribution (isd)
        nS = 5
        env = gridworlds.EnvGridworld1D(length=nS, rewards_dict={nS-1: +1}, reward_default=0.0)
        env.setStoreTrajectoryFlag(True)
        absorption_set = set(range(1))
        absorption_boundary = max(absorption_set)
        activation_set = {absorption_boundary + 1}
        active_set = set(np.arange(absorption_boundary + 1, env.getNumStates()))
        print(f"Absorption set: {absorption_set}")
        print(f"Activation set: {activation_set}")
        learner_fv = LeaFV(env, N=3, T=10, absorption_set=absorption_set, activation_set=activation_set, criterion=LearningCriterion.DISCOUNTED, gamma=0.9)

        # Now we mock the evolution of the 3 particles in the FV particle system
        # We do so by creating a data frame index by each time the particle system is updated, i.e. indexed by the FV system's clock
        # The particle that is updated at each time is defined by a separate list containing that information
        df_trajectories = pd.DataFrame({0: [2, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 3, 2],
                                        1: [3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        2: [2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4]})
        particle_to_update = [0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 0, 0, -1]
        assert len(particle_to_update) == len(df_trajectories)
        print(pd.concat([df_trajectories, pd.DataFrame({'particle_to_update': particle_to_update})], axis=1))

        # Particle system evolution based on the df_trajectories data frame
        # Note that the FIRST row in each particle's trajectory corresponds to the first time at which the particle is moved
        # The initial state of each particle is thus the value of 'state' at the first row.
        # Particles, whose trajectories will evolve as per the df_trajectories data frame just created
        particles = [copy.deepcopy(learner_fv.env) for _ in range(learner_fv.getNumParticles())]
        # Store the start states at the first row (so that we know at which state each particle is at the onset of the experiment)
        start_states = [df_trajectories[p].iloc[0] for p in range(learner_fv.getNumParticles())]

        for p, _particle in enumerate(particles):
            _particle.setState(start_states[p])
            _particle.reset_trajectory()
            assert np.allclose(_particle.getTrajectory(), [0, start_states[p], -1, start_states[p], env.getReward(start_states[p])], equal_nan=True), \
                "At the onset of the experiment, each particle's trajectory must have only ONE record having its state and next_state equal to their start state, and action equal NaN"
        for t, p in enumerate(particle_to_update[:-1]):  # We remove the last element in particle_to_update because at the last record of the data frame no particle should be updated because trajectories have reached to an end
            print(f"\ntime = {t+1}: Particle #{p} MOVES")
            state = df_trajectories.iloc[t, p]
            action = int( 0.5*(df_trajectories.iloc[t+1, p] - df_trajectories.iloc[t, p]) + 0.5 )
            next_state = df_trajectories.iloc[t+1, p]
            reward = env.getReward(next_state)
            particles[p].update_trajectory(t+1, state, action, next_state, reward)  # We store t+1 because t starts at 0, which is the time at which the particle is at its initial state, and in the particle's trajectory we store the time at which the particle transitions to the NEXT state (i.e. the trajectory is continuous from the right)
            for pp, _particle in enumerate(particles):
                print(f"Particle #{pp}:\n{_particle.getTrajectory()}")

        # Update the value of Phi(t,x) for each state of interest x defined in the FV learner (which by default is the set of terminal states of the environment)
        learner_fv.update_phi_function(particles)
        for x in learner_fv.states_of_interest:
            print(f"\n*** Phi(t, x={x})***")
            for s in sorted(learner_fv.dict_phi_for_state.keys()):
                print(f"s = {s}:")
                print(learner_fv.dict_phi_for_state[s][x])
                for a in range(env.getNumActions()):
                    print(f"s = {s}, a = {a}:")
                    print(learner_fv.dict_phi_for_state_action[s][a][x])
                print("")

        # Check expected results
        expected_dict_phi_for_state = dict({1: {4: pd.DataFrame({'t': np.arange(6), 'Phi': [0, 0, 0, 1, 0, 0]}, columns=['t', 'Phi'])},
                                            2: {4: pd.DataFrame({'t': np.arange(7), 'Phi': [0, 0, 0.666667, 0, 0.666667, 0, 0]}, columns=['t', 'Phi'])},
                                            3: {4: pd.DataFrame({'t': np.arange(4), 'Phi': [0, 0.6, 0, 0.5]}, columns=['t', 'Phi'])},
                                            4: {4: pd.DataFrame({'t': np.arange(3), 'Phi': [1, 0, 0.5]}, columns=['t', 'Phi'])}
                                        })
        assert learner_fv.dict_phi_for_state.keys() == learner_fv.active_set
        assert learner_fv.dict_phi_for_state.keys() == expected_dict_phi_for_state.keys()
        for s in learner_fv.active_set:
            assert list(learner_fv.dict_phi_for_state[s].keys()) == learner_fv.states_of_interest
            for x in learner_fv.states_of_interest:
                assert np.allclose(learner_fv.dict_phi_for_state[s][x], expected_dict_phi_for_state[s][x])
