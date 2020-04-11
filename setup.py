# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:23:01 2020

@author: Daniel Mastropietro
@description: Setup the projects' environment (e.g. make the projects portable)
"""

def add_current_dir_to_path():
    import os
    import sys
    #print("Currently in file: " + __file__)
    PATH_TO_FILE = os.path.abspath(__file__)
    ROOTPATH = os.path.dirname(PATH_TO_FILE)
    try:
        sys.path.index(ROOTPATH)
    except:
        sys.path.insert(0, ROOTPATH)
        print("Directory:\n" + ROOTPATH + "\nhas been prepended to the module search path.")


add_current_dir_to_path()
