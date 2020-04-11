# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:48:52 2020

@author: Daniel Mastropietro
@description: Test the gridworld environment with different learning algorithms.
"""

if __name__ == "__main__":
    # Setup the working environment so that e.g. modules defined in 'Python/lib' are found
    import runpy
    runpy.run_path('../setup.py')

else:
    import Python.lib.gridworld as gridworld
    from Python.lib.GitHub.MJeremy2017 import RL
    
    env_grid = gridworld.GridworldEnv()
    env_grid._render()
