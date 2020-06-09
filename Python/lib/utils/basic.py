# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 21:43:52 2020

@author: Daniel Mastropietro
@description: Functions used to help in basic operations that are usually complicated in Python
"""

def find(alist, value):
    """
    Returns all the indices in `alist` that are equal to `value`.

    Arguments:
    alist: list
        List where `value` is searched for.
    value: any object
        A value/object to search in `alist`.

    Return: list
        A list containing all the indices where the list element is equal to `value`. 
    """

    return [i for i, v in enumerate(alist) if v == value]
