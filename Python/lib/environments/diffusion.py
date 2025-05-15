# -*- coding: utf-8 -*-
"""
Created on 08 May 2025

@author: Daniel Mastropietro
@description: Definition of continuous-time continuous-state environments defined by diffusion processes, such as the Ornstein-Uhlenbeck process.
                The environment is simulated by doing a time discretization.
                The environment must define a step() method that returns the next state given the new state, and a reward associated to such transition.
@notes: The file also defines function callables that are used to define the drift and the noise of the stochastic process.
        All these callables should receive the state as input parameter and optional keyword parameters used to compute the returned values.
"""

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../../setup.py')

import numpy as np
# Consider using jit (Just-In-Time) compilation to significantly accelerate execution processes
# (e.g. 10x compared to not using jit, as observed by switching the option in vu_frequencyprocesssimulation.py)
# However, note that using @jit as decorator of a CLASS brings some issues and that is why another decorator called @jitclass was created (but still the process doesn't work)
#from numba import jit, jitclass
## For jit see: https://www.geeksforgeeks.org/numpy-optimization-with-numba/
## For jitclass see: https://numba.pydata.org/numba-doc/dev/user/jitclass.html

from Python.lib.environments import EnvironmentContinuous
from Python.lib.utils.basic import keep_dict_params_defined_in_function


def func_drift_const(state, theta=1.0):
    "Function that defines a constant drift that does not depend on the state of the diffusion process"
    return theta


#@jit
def func_drift_fcr_piecewise(Ft, r=0.6, x=1.4, mu=0.0):
    """Calculate the state-dependent drift function based on the given piecewise function

    Ref: arXiv:2411.11093, "Dynamic Dimensioning of Frequency Containment Reserves: The Case of the Nordic Grid", Janssen, Zocca, Zwart, Kazempour (2024)

    Parameters:
    - Ft: Current value of the process.
    - r: Parameter for the drift function due to the FCR-N capacity.
    - x: Parameter for the drift function due to the FCR-D capacity.
    - mu: Long-term mean of the process.

    Returns:
    - F: The drift value based on the piecewise function.
    """
    F = Ft - mu
    if F < -0.5 or F > 0.5:
        alpha = r + x
    elif -0.5 <= F <= -0.1:
        alpha = r - 0.25 * x - 2.5 * x * F
    elif -0.1 < F <= 0.0:
        alpha = -10 * r * F
    elif 0.0 < F <= 0.1:
        alpha = 10 * r * F
    elif 0.1 < F <= 0.5:
        alpha = r - 0.25 * x + 2.5 * x * F
    else:
        raise ValueError("F(t) is out of the expected range")

    # We return a drift that is mean reverting
    return -alpha


def func_noise_gaussian(_state, sigma=1.0, dt=1.0):
    """
    Zero-mean Gaussian noise that is independent of the state x representing a realization of a discrete-time Brownian motion
    whose variance is proportional to the square root of the discrete time interval, i.e. equal to sigma*sqrt(dt).
    """
    return np.random.normal(0, sigma*np.sqrt(dt))


# I think we need to define the class with @jitclass if we want to use @jit for the simulation
#@jitclass   # See https://numba.pydata.org/numba-doc/dev/user/jitclass.html
class EnvDiffusion(EnvironmentContinuous):
    """
    Environment whose dynamics are governed by a discretized version of a stochastic differential equation such as e.g. the Ornstein-Uhlenbeck (OU) process

    Ref for OU process: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Arguments:
    func_drift: callable
        Function that returns the drift of the process as a function of the state and other parameters passed as kwargs.
        default: func_drift_const, a function that is constant, independent of the state of the process

    func_drift: callable
        Function that returns the noise process as a function of the state and other parameters passed as kwargs.
        default: func_noise_gaussian, which is a zero-mean Gaussian process independent of the state

    mu: float
        Mean of the OU process.
        default: 0.0

    sigma: positive float
        Standard deviation of the Gaussian noise.
        default: 0.1

    dt: positive float
        Time discretization interval of the continuous-time OU process.
        default: 0.1

    **kwargs: additional parameters needed for the simulation of the process.
    These parameters are passed to the functions defined in `func_drift` and `func_noise`.
    Note that parameters `mu`, `sigma`, `dt` passed to the constructor are added to the dictionary storing these kwargs parameters,
    as they are commonly used by those functions.
    """
    def __init__(self,  func_drift: callable=func_drift_const, func_noise: callable=func_noise_gaussian,
                        mu=0.0, sigma=0.1, dt=0.1, **kwargs):
        super().__init__()
        self.func_drift = func_drift
        self.func_noise = func_noise
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.dict_params = kwargs       # Parameters used in methods of the class (e.g. by the drift function callable)

        # Add mu, sigma, dt to the dict_params dictionary because they are usually used for the drift and noise functions
        # This is needed because it is NOT possible to pass a parameter twice (in this case, pass `mu`, `sigma` and 'dt' ALSO as part of the **kwargs parameter)
        if 'mu' not in self.dict_params.keys():
            self.dict_params['mu'] = mu
        if 'sigma' not in self.dict_params.keys():
            self.dict_params['sigma'] = sigma
        if 'dt' not in self.dict_params.keys():
            self.dict_params['dt'] = dt

    def reset(self):
        "Resets the state to the long-run average"
        self.state = self.mu

    def step(self, action):
        "Environment step method defining the next state given the current state and an action. It should return the same information returned by the gym step() methods"
        # Compute next state based on drift and noise terms
        next_state = self.state + self.drift() + self.noise()
        reward = 0.0
        done = False
        info = {}

        # Update the state of the system
        self.setState(next_state)

        return next_state, reward, done, info

    def drift(self):
        "Returns a realization of the drift term of the stochastic differential equation"
        params_func = keep_dict_params_defined_in_function(self.dict_params, self.func_drift)
        return self.func_drift(self.state, **params_func) * (self.state - self.mu) * self.dt

    def noise(self):
        "Returns a realization of the noise term of the stochastic differential equation"
        params_func = keep_dict_params_defined_in_function(self.dict_params, self.func_noise)
        return self.func_noise(self.state, **params_func)

    def render(self, mode='human'):
        pass

    def setState(self, state):
        "Sets the state of the environment"
        self.state = state

    def getState(self):
        return self.state
