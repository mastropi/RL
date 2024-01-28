# -*- coding: utf-8 -*-
"""
Created on Mon Sep 4 16:32:28 2023

@author: Daniel Mastropietro
@description: Support functions used in Fleming-Viot simulations for both discrete-time and continuous-time contexts.
They are expected to be applicable to:
- queue environments
- discrete-time/state/action environments
"""

from enum import Enum, unique

import numpy as np

@unique
class ReactivateMethod(Enum):
    RANDOM = 1                  # Random choice of the reactivation particle among the N-1 other non-absorbed particles
    VALUE_FUNCTION = 2          # Choice of the reactivation particle based on the value function of each state at which the other N-1 particles are located
    ROBINS = 3


def reactivate_particle(envs: list, idx_particle: int, K: int, reactivation_number: int=None, all_idx_particles: list=None):
    """
    Reactivates a particle that has been absorbed

    Reactivation means that the state of the particle is set to be the state of the chosen particle.

    Arguments:
    envs: list
        List of environments used to run the FV process. It is used to retrieve the state of the reactivation particle and to set the new state
        of the absorbed particle `idx_particle`.

    idx_particle: int
        The index of the particle to be reactivated. It must be a valid index in the `envs` list.

    K: int
        For queue environments, this is the queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION and only makes sense for queue environments.
        It is used in the call to choose_particle() done by this function.

    reactivation_number: (opt) int
        Counter of the number of absorptions that took place until the current absorption being dealt with that is used to choose the reactivation particle.
        When given, ReactivateMethod.ROBINS is used as reactivation method.
        default: None

    all_idx_particles: (opt) list
        List containing the particle IDs among which the reactivation particle should be chosen.
        default: None

    Return: int
    The index of the particle chosen for reactivation. This value indexes the envs list.
    """
    # Select a reactivation particle out of the other N-1 particles
    if all_idx_particles is None:
        N = len(envs)
    else:
        assert len(all_idx_particles) <= len(envs)
        N = len(all_idx_particles)
    assert N > 1, "There is more than one particle in the system (N={})".format(N)

    reactivate_method = ReactivateMethod.RANDOM
    if reactivation_number is not None and reactivation_number >= 0:
        reactivate_method = ReactivateMethod.ROBINS
        # Make sure that the reactivation number is between 0 and N-2 because N may NOT include ALL particles in envs, it might be smaller when parameter all_idx_particles is not None
        # If the condition is already true, the following line will NOT change its value.
        reactivation_number = reactivation_number % (N - 1)
    idx_reactivate_out_of_N = choose_particle(envs, idx_particle, N, K, reactivate_method, reactivation_number=reactivation_number) #ReactivateMethod.VALUE_FUNCTION)

    if all_idx_particles is None:
        # The number of elements in envs is N
        # => The value of idx_reactivate_out_of_N can be readily used to index (through envs) the particle that was chosen for reactivation
        idx_reactivate = idx_reactivate_out_of_N
    else:
        # The number of elements in envs is >= N
        # => Compute the index in envs that represents the particle that was chosen for reactivation
        # Ex: all_idx_particles = [1, 3, 4]
        # whereas envs is made up of 6 particles, in this case N = len(all_idx_particles) = 3, and since idx_reactivate_out_of_N takes a value between
        # 0 and N-1, in this case it takes either the value 0, 1 or 2.
        # Therefore the index representing the particle that was chosen for reactivation is given by all_idx_particles[idx_reactivate_out_of_N]
        idx_reactivate = all_idx_particles[idx_reactivate_out_of_N]

    # Update the state of the reactivated particle to the reactivation state
    envs[idx_particle].setState(envs[idx_reactivate].getState())

    return idx_reactivate


def choose_particle(envs: list, idx_particle: int, N: int, K: int, method: ReactivateMethod, reactivation_number: int=None):
    """
    Chooses a particle for reactivation of an absorbed particle among N-1 possible particles using the given reactivation method

    Arguments:
    envs: list
        List of environments used to run the FV process. Only used when method = ReactivateMethod.VALUE_FUNCTION.

    idx_particle: int
        The index of the particle to be reactivated.

    N: int
        Number of particles in the Fleming-Viot system.

    K: int
        For queue environments, this is the queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION and only makes sense for queue environments.

    method: ReactivateMethod
        Reactivation method to use which defines how the reactivation particle is chosen
        (e.g. randomly, in order, or using a probability distribution that depends on the value function of the state
        where each potential non-absorbed particle is located).

    reactivation_number: (opt) int
        Counter of the number of absorptions that took place until the current absorption being dealt with that is used to choose the reactivation particle
        when the reactivation method is ReactivateMethod.ROBINS.
        When given, it should be an integer between 0 and N-2, as the reactivation particle needs to be chosen among N-1 particles.
        default: None

    Return: int
    The index of the chosen particle.
    """
    if method == ReactivateMethod.VALUE_FUNCTION:
        # TODO: (2022/03/10) Inquiry the value function of each of the N-1 particles based on their state and use a probability distribution that is proportional to it.
        # To begin with, consider an increasing linear function of the buffer size, and on a next step estimate the value function
        # appropriately based on the environment's dynamics.
        # Note: the value function should be defined as part of the learner in self.learnerV attribute of the SimulatorQueue object.

        # In queue environments, the value function is considered a linear function of the buffer size.
        # Note that the length of `values` is equal to the number of particles in the system minus 1, N-1
        values = [envs[idx].getBufferSize() / K for idx in list(range(idx_particle)) + list(range(idx_particle + 1, N))]
        assert len(values) == N - 1
        prob_values = [v / np.sum(values) for v in values]
        # print(np.c_[range(N-1), [envs[idx].getBufferSize() for idx in range(N-1)], [envs[idx].getBufferSize() / K for idx in range(N-1)], prob_values])
        idx_reactivate = np.random.choice(N-1, p=prob_values)
    elif method == ReactivateMethod.ROBINS:
        # Deterministic choice of the particle
        assert reactivation_number is not None
        assert 0 <= reactivation_number < N - 1, "The reactivation number is between 0 and N-2 = {} ({})".format(N-2, reactivation_number)
        idx_reactivate = reactivation_number
    else:
        # Random selection of active particles by default
        idx_reactivate = np.random.randint(0, N-1)  # The upper value is NOT included in the possible set of integers

    # Make the chosen particle index satisfy the condition that is a number between 0 and N-1, excluding `idx_particle`
    # Recall that above, the reactivate index is chosen between 0 and N-2, hence, if the chosen index is >= idx_particle
    # we should increase the index by 1 so that it satisfies the condition just mentioned.
    if idx_reactivate >= idx_particle:
        idx_reactivate += 1
    assert 0 <= idx_reactivate < N, "The reactivation particle ID ({}) is between 0 and N (={}) for particle with ID = {}" \
        .format(idx_reactivate, N, idx_particle)
    assert idx_reactivate != idx_particle, "The particle chosen for reactivation ({}) is different from the particle being reactivated ({})" \
        .format(idx_reactivate, idx_particle)

    # print("Reactivated particle ID for particle `{}` and its buffer size: {}, {} (particles @: {})"
    #      .format(idx_particle, idx_reactivate, envs[idx_reactivate].getBufferSize(), [env.getBufferSize() for env in envs]))

    return idx_reactivate


