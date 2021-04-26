# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 21:43:52 2020

@author: Daniel Mastropietro
@description: Functions used to help in basic operations that are usually complicated in Python
"""

import bisect   # To insert elements in order in a list (bisect.insort(list, element)
import copy
import numpy as np

def array_of_objects(size, value=None, dtype=list):
    """
    Creates an array of objects of the specified type.
    Each element of the array is optionally filled to a given value, the same for all elements.
    
    Arguments:
    size: int, tuple
        Size of the array, which is used when calling numpy.empty().

    value: (optional)
        Value to assign to each element of the array when it is created.
        Note that a COPY of it is assigned so that they don't share the same memory address. 
        default: None

    dtype: (optional)
        Type of object to be stored in each element of the array.
        Its value is used as the `dtype` parameter of the numpy.empty() call that is used to create the array.
        default: list

    Return: numpy array
    Numpy array of size `size` containing `value` as each element's value if different from `None`.
    """
    arr = np.empty(size, dtype=dtype)

    # Fill in the values of the array if `value` is not None 
    if value is not None:
        if len(size) == 1:
            # 1D array
            for idx in range(len(arr)):
                # We store a COPY of `value` so that in case the value is immutable (e.g. a list)
                # they don't share the same memory address and they can be changed without affecting the other entries of the array!
                arr[idx] = copy.deepcopy(value)
        else:
            # Multi-dimensional array
            # Fill the values using a stacked version of the array
            # (so that the array can be filled regardless of its shape)
            arr_stacked = arr.reshape( np.prod(size) )
            for idx in range(len(arr_stacked)):
                # We store a COPY of `value` so that in case the value is immutable (e.g. a list)
                # they don't share the same memory address and they can be changed without affecting the other entries of the array!
                arr_stacked[idx] = copy.deepcopy(value)
            arr = arr_stacked.reshape(size)

    return arr

def insort(alist, value, unique=False):
    """
    Inserts a value in a list in order and returns the index of insertion
    together with a flag stating whether the value was already present in the list. 
    
    The value is not inserted if unique=True and the value already existed in the list.

    Arguments:
    unique: bool
        Whether the values in `alist` should be unique.
        If True, `value` is NOT inserted in `alist`,
        but the index where it would have been inserted is returned.

    Return: tuple
    The tuple contains two elements:
    - the index of the sorted insertion if unique=False or if the value does not already
    exist in the list, or the index of the last occurrence of `value` in the input list
    otherwise.
    - a flag stating whether the value already existed in the list 
    """
    # Index where to insert the value respecting the list order
    # NOTE that, when the value is already in the list, bisect()
    # returns an index number that is ONE more the index where
    # the value is found in the list.
    idx = bisect.bisect(alist, value)
    if idx > 0 and alist[idx-1] == value:
        found = True
    else:
        found = False

    if not unique or \
       unique and not found:
        # Insert the value respecting the list order
        alist.insert(idx, value)

    if unique and found:
        # The returned index is the index of the last occurrence of
        # the element already existing in the list, so that this
        # index can be used directly as an index "to update" in case
        # we need to update another list that is related to the list
        # given as input to this function.
        idx = idx - 1

    return idx, found

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

def find_first(alist, value):
    """
    Returns the smallest index in `alist` that is equal to `value`

    Arguments:
    alist: list
        List where `value` is searched for.
    value: any object
        A value/object to search in `alist`.

    Return: list
        A list containing the smallest index where the list element is equal to `value`
        or -1 if `value` is not present.
    """
    for i, v in enumerate(alist):
        if v == value:
            return i
    return -1

def find_last(alist, value):
    """
    Returns the smallest index in `alist` that is equal to `value`

    Arguments:
    alist: list
        List where `value` is searched for.
    value: any object
        A value/object to search in `alist`.

    Return: list
        A list containing the largest index where the list element is equal to `value`
        or -1 if `value` is not present.
    """
    for i in range(len(alist)-1, -1, -1):
        if alist[i] == value:
            return i
    return -1

def merge_values_in_time(t1, y1, t2, y2, unique=False):
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

    unique: bool
        Whether the merged t values should be unique while merging unique values in input t1 and t2.
        If repeated values are found in the input time lists t1 and t2, the respective y1 or y2 value
        associated to the index of the LAST occurrence of the repeated time value is kept.

    Return: tuple
    - List of merged times
    - List with the first list of measurements resulting after merge, possibly after cleaning up
    the respective time list of repeated time values, and keeping the last occurrence.
    - List with the second list of measurements resulting after merge, possibly after cleaning up
    the respective time list of repeated time values, and keeping the last occurrence.
    """

    #-------- Parse input parameters
    if not isinstance(t1, list) or not isinstance(t2, list) or not isinstance(y1, list) or not isinstance(y2, list):
        raise ValueError("The time and measurement values (ti, yi) should be given as list. They are of type: t1: {}, t2: {}, y1: {}, y2: {}".format(type(t1), type(t2), type(y1), type(y2)))
    if len(t1) != len(y1) or len(t2) != len(y2):
        raise ValueError("The time values (ti) should have the same length as their respective measurements (yi)")
    if t1[0] != 0 or t2[0] != 0:
        raise ValueError("The first value in each list of times must be 0.")
    if sorted(t1) != t1 or sorted(t2) != t2:
        raise ValueError("At least one of the time lists is not sorted increasingly.\nt1={}\nt2={}".format(t1, t2))

    if unique:
        # Remove duplicates from each input time list to merge
        # For each measurement series y, the y value for the FIRST occurrence of the repeated time value is kept 
        t1_merged, idx1 = np.unique(t1, return_index=True)
        t1_merged = list(t1_merged)
        idx1_next = list(idx1[1:]) + list([ idx1[-1]+1 ])
        y1_merged = [y1[i-1] for i in idx1_next]

        t2_merged, idx2 = list( np.unique(t2, return_index=True) )
        t2_merged = list(t2_merged)
        idx2_next = list(idx2[1:]) + list([ idx2[-1]+1 ])
        y2_merged = [y2[i-1] for i in idx2_next]
    else:
        t1_merged = t1.copy()
        y1_merged = y1.copy()
        t2_merged = t2.copy()
        y2_merged = y2.copy()

    # Insert values in the first pair of lists
    for t in t2:
        if (t > 0):
            idx_insort, found = insort(t1_merged, t, unique=unique)
            assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
            if not unique or not found:
                y1_merged.insert(idx_insort, y1_merged[idx_insort-1])

    # Insert values in the second pair of lists
    for t in t1:
        if (t > 0):
            idx_insort, found = insort(t2_merged, t, unique=unique)
            assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
            if not unique or not found:
                y2_merged.insert(idx_insort, y2_merged[idx_insort-1])

    assert len(t1_merged) == len(y1_merged) == len(t2_merged) == len(y2_merged), \
            "The length of the first pair of merged lists is the same" \
            " as that of the second pair of merged lists ({}, {}, {}, {})" \
            .format(len(t1_merged), len(y1_merged), len(t2_merged), len(y2_merged))
    assert t1_merged == t2_merged, \
            "The times lists resulting from the merge are the same ({}, {})" \
            .format(t1_merged, t2_merged)

    return t1_merged, y1_merged, y2_merged


if __name__ == "__main__":
    #-------------------- merge_values_in_time ------------------------#
    print("\n--- merge_values_in_time(): Test #1 on unique time values across series")
    t1 = [0.0, 2.5, 3.2, 7.2, 11.3]
    y1 = [  4,   3,   2,   1,    0]
    t2 = [0.0, 1.1, 2.51, 2.87, 3.3, 4.8, 6.9]
    y2 = [  0,   0,    1,    2,   1,   1,   0]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=True)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.51, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3]
    assert y1f == [4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 0]
    assert y2f == [0, 0, 0, 1, 2, 2, 1, 1, 0, 0, 0]   

    #-------------------------
    print("\n--- merge_values_in_time(): Test #2 on UNIQUE time values WITHIN series but some repeated time values across series (UNIQUE=True)")
    t1 = [0.0, 2.5, 3.2, 7.2, 11.3]
    y1 = [  4,   3,   2,   1,    0]
    t2 = [0.0, 1.1, 2.50, 2.87, 3.3, 4.8, 6.9]  # The difference with Test #1 data is here, at element idx=2 which is now 2.5 instead of 2.51
    y2 = [  0,   0,    1,    2,   1,   1,   0]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=True)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3]
    assert y1f == [4, 4, 3, 3, 2, 2, 2, 2, 1, 0]
    assert y2f == [0, 0, 1, 2, 2, 1, 1, 0, 0, 0]

    #-------------------------
    print("\n--- merge_values_in_time(): Test #3 on UNIQUE time values WITHIN series but some repeated time values across series (UNIQUE=False)")
    t1 = [0.0, 2.5, 3.2, 7.2, 11.3]
    y1 = [  4,   3,   2,   1,    0]
    t2 = [0.0, 1.1, 2.50, 2.87, 3.3, 4.8, 6.9]
    y2 = [  0,   0,    1,    2,   1,   1,   0]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=False)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.5, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3]
    assert y1f == [4,   4,   3,   3,    3,   2,   2,   2,   2,   1,    0]
    assert y2f == [0,   0,   1,   1,    2,   2,   1,   1,   0,   0,    0]

    #-------------------------
    print("\n--- merge_values_in_time(): Test #4 on REPEATED time values WITHIN series and some repeated time values across series (UNIQUE=True)")
    t1 = [0.0, 2.5, 3.2, 3.2, 3.2, 7.2, 11.3]   # 3.2 is repeated 3 times
    y1 = [  4,   3,   4,   5,   6,   5,    4]
    t2 = [0.0, 1.1, 2.50, 2.87, 3.3, 4.8, 6.9]
    y2 = [  0,   0,    1,    2,   1,   1,   0]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=True)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3]
    assert y1f == [4,   4,   3,    3,   6,   6,   6,   6,   5,    4]
    assert y2f == [0,   0,   1,    2,   2,   1,   1,   0,   0,    0]

    #-------------------------
    print("\n--- merge_values_in_time(): Test #5 same as Test #4 but with Unique=False")
    t1 = [0.0, 2.5, 3.2, 3.2, 7.2, 11.3]
    y1 = [  4,   3,   4,   5,   4,    3]
    t2 = [0.0, 1.1, 2.50, 2.87, 3.3, 4.8, 6.9]  # The difference with Test #1 data is here, at element idx=2 which is now 2.5 instead of 2.51
    y2 = [  0,   0,    1,    2,   1,   1,   0]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=False)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.5, 2.87, 3.2, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3]
    assert y1f == [4,   4,   3,   3,    3,   4,   5,   5,   5,   5,   4,    3]
    assert y2f == [0,   0,   1,   1,    2,   2,   2,   1,   1,   0,   0,    0]
    #-------------------- merge_values_in_time ------------------------#
