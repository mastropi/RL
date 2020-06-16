# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 21:43:52 2020

@author: Daniel Mastropietro
@description: Functions used to help in basic operations that are usually complicated in Python
"""

import bisect   # To insert elements in order in a list (bisect.insort(list, element)

def insort(alist, value):
    """
    Inserts a value in a list in order and returns the index of insertion
    (something that the bisect.insort() function does not do)
     """
    idx = bisect.bisect(alist, value)  # Index where to insert the value respecting the list order
    alist.insert(idx, value)           # Insert the new value respecting the list order
    return idx

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

def merge_values_in_time(t1, y1, t2, y2):
    """
    Merges two list of values measured at different times into two separate lists
    of values measured both at the same times.

    ASSUMPTIONS:
    - The length of each time list (ti) is the same as the list of respective measurements (yi).
    - The two time lists have only ONE value in common, namely t = 0, which is also the smallest time value.
    - The two time lists are sorted increasingly.

    Arguments:
    t1: list
    List of times associated to the first list of measurements y1, assumed sorted increasingly. 

    y1: list
    List of measurements associated to the first times list t1.

    t2: list
    List of times associated to the second list of measurements y2, assumed sorted increasingly. 

    y2: list
    List of measurements associated to the first times list t1.

    Return: tuple
    - List of merged times
    - List with the first list of measurements resulting after merge 
    - List with the second list of measurements resulting after merge 
    """

    #-------- Parse input parameters
    if len(t1) != len(y1) or len(t2) != len(y2):
        raise ValueError("The time values (ti) should have the same length as their respective measurements (yi)")
    if t1[0] != 0 or t2[0] != 0:
        raise ValueError("The first value in each list of times must be 0.")
    if sorted(t1) != t1 or sorted(t2) != t2:
        raise ValueError("At least one of the time lists is not sorted increasingly.")

    t1_merged = t1.copy()
    y1_merged = y1.copy()
    t2_merged = t2.copy()
    y2_merged = y2.copy()

    # Insert values in the first pair of lists
    for t in t2:
        if (t > 0):
            idx_insort = insort(t1_merged, t)
            assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
            y1_merged.insert(idx_insort, y1_merged[idx_insort-1])

    # Insert values in the second pair of lists
    for t in t1:
        if (t > 0):
            idx_insort = insort(t2_merged, t)
            assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
            y2_merged.insert(idx_insort, y2_merged[idx_insort-1])

    assert len(t1_merged) == len(y1_merged) == len(t2_merged) == len(y2_merged), \
            "The length of the first pair of merged lists is the same" \
            " as that of the second pair of merged lists ({}, {}, {}, {})" \
            .format(len(t1_merged), len(y1_merged), len(t2_merged), len(y2_merged))
    assert t1_merged == t2_merged, \
            "The times lists resulting from the merge are the same ({}, {})" \
            .format(t1_merged, t2_merged)

    return t1_merged, y1_merged, y2_merged
