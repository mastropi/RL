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


def reactivate_particle(envs: list, idx_particle: int, K: int, absorption_number=None):
    """
    Reactivates a particle that has been absorbed

    Reactivation means that the state of the particle is set to be the state of the chosen particle.

    Arguments:
    envs: list
        List of queue environments used to run the FV process.

    idx_particle: int
        The index of the particle to be reactivated.

    K: int
        Queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION is used in the call to
        choose_particle() done by this function.

    Return: int
    The index of the particle to which it is reactivated.
    """
    # Select a reactivation particle out of the other N-1 particles
    N = len(envs)
    assert N > 1, "There is more than one particle in the system (N={})".format(N)
    idx_reactivate = choose_particle(envs, idx_particle, N, K, ReactivateMethod.RANDOM)  # , ReactivateMethod.VALUE_FUNCTION) #, ReactivateMethod.RANDOM)
    # idx_reactivate = choose_particle(envs, idx_particle, N, K, ReactivateMethod.ROBINS, absorption_number=absorption_number) #, ReactivateMethod.VALUE_FUNCTION) #, ReactivateMethod.RANDOM)

    # Update the state of the reactivated particle to the reactivation state
    envs[idx_particle].setState(envs[idx_reactivate].getState())

    return idx_reactivate


def choose_particle(envs: list, idx_particle: int, N: int, K: int, method: ReactivateMethod, absorption_number=None):
    """
    Chooses a particle among N-1 possible particles using the given method

    Arguments:
    envs: list
        List of queue environments used to run the FV process. Only used when method = ReactivateMethod.VALUE_FUNCTION.

    idx_particle: int
        The index of the particle to be reactivated.

    N: int
        Number of particles in the Fleming-Viot system.

    K: int
        Queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION.

    method: ReactivateMethod
        Reactivation method to use which defines how the particle is chosen
        (e.g. randomly or using a probability distribution that depends on the value function of the state
        where each potential non-absorbed particle is located).

    Return: int
    The index of the chosen particle.
    """
    if method == ReactivateMethod.VALUE_FUNCTION:
        # TODO: (2022/03/10) Inquiry the value function of each of the N-1 particles based on their state and use a probability distribution that is proportional to it.
        # To begin with, consider an increasing linear function of the buffer size, and on a next step estimate the value function
        # appropriately based on the queue's dynamics.
        # Note: the value function should be defined as part of the learner in self.learnerV attribute of the SimulatorQueue object.

        # The value function is a linear function of the buffer size
        # Note that the length of `values` is equal to the number of particles in the system minus 1, N-1
        values = [envs[idx].getBufferSize() / K for idx in list(range(idx_particle)) + list(range(idx_particle + 1, N))]
        assert len(values) == N - 1
        prob_values = [v / np.sum(values) for v in values]
        # print(np.c_[range(N-1), [envs[idx].getBufferSize() for idx in range(N-1)], [envs[idx].getBufferSize() / K for idx in range(N-1)], prob_values])
        idx_reactivate = np.random.choice(N - 1, p=prob_values)
    elif method == ReactivateMethod.ROBINS:
        # Deterministic choice of the particle
        assert absorption_number is not None
        assert 0 <= absorption_number < N - 1, "The absorption number is between 0 and N-2 = {} ({})".format(N - 2, absorption_number)
        idx_reactivate = absorption_number
    else:
        # Random selection of active particles by default
        idx_reactivate = np.random.randint(0, N - 1)  # The upper value is NOT included in the possible set of integers

    if idx_reactivate >= idx_particle:
        # The chosen particle is beyond the particle to reactivate
        # => increase the index of the particle by 1 so that we choose the correct particle
        idx_reactivate += 1
    assert 0 <= idx_reactivate < N, "The reactivation particle ID ({}) is between 0 and N (={}) for particle with ID = {}" \
        .format(idx_reactivate, N, idx_particle)
    assert idx_reactivate != idx_particle, "The particle chosen for reactivation ({}) is different from the particle being reactivated ({})" \
        .format(idx_reactivate, idx_particle)

    # print("Reactivated particle ID for particle `{}` and its buffer size: {}, {} (particles @: {})"
    #      .format(idx_particle, idx_reactivate, envs[idx_reactivate].getBufferSize(), [env.getBufferSize() for env in envs]))

    return idx_reactivate


