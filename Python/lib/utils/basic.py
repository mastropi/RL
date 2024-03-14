# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 21:43:52 2020

@author: Daniel Mastropietro
@description: Functions used to help in basic operations that are usually complicated in Python
"""

import bisect   # To insert elements in order in a list (bisect.insort(list, element)
import copy
import re
import joblib
import numpy as np
import pandas as pd

import datetime
from time import process_time
from timeit import default_timer as timer
from unittest import TestCase


# Decorator to measure execution time
# (Inspired from uugot.it project -> utils.measure_exec_time())
def measure_exec_time(func):
    """
    Decorator that measures the execution time of the decorated function
    and prints it as long as it is larger than 1 second.

    Note: for redirection of output to a logger object, see the aforementioned function in uugot.it project.

    Ex:
    @measure_exec_time
    def fun(*args)
        ...
        return results
    results = fun() # This will print the execution time of the function in the standard output.
    """
    def func_decorated(*args, **kwargs):
        start_time = timer()
        start_cpu = process_time()
        results = func(*args, **kwargs)
        end_time = timer()
        end_cpu = process_time()
        exec_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        #if "{:.1f}".format(exec_time) != "0.0":
        if exec_time > 0: #> 1
            print("+++ Execution time for {}: {:.1f} hs = {:.1f} min = {:.1f} sec = {:.2f} msec (CPU: {:.1f} hs, {:.1f} min, {:.1f} sec, {:.2f} msec)" \
                  .format(func.__name__, exec_time/3600, exec_time/60, exec_time, exec_time*1000,
                          cpu_time/3600,cpu_time/60, cpu_time, cpu_time*1000))
        return results
    return func_decorated

def get_exception_message(e):
    return getattr(e, "message", str(e))

def get_current_datetime_as_string(format=None):
    """
    Returns the current time as a datetime string using the current datetime.

    It is the value returned by `datetime.datetime.today().strftime()`.

    Arguments:
    format: str or None
        Format hint for the output datetime string.
        When "filename" a datetime string suitable to be used in a filename is generated in the form "YYYYMMDD_HHMMSS".
        Otherwise the following format is used: "%Y-%m-%d %H:%M:%S".

    Return: str
    Current datetime as a string.
    """
    if format is "filename":
        format_strftime = "%Y%m%d_%H%M%S"
    else:
        format_strftime = "%Y-%m-%d %H:%M:%S"

    return generate_datetime_string(format=format_strftime)

def get_datetime_from_string(dt, format="%Y-%m-%d %H:%M:%S"):
    "Returns the datetime associated to a string in the given format"
    return datetime.datetime.strptime(dt, format)

def generate_datetime_string(dt=None, format="%Y%m%d_%H%M%S", prefix="", suffix="", extension="", sep="_"):
    """
    Returns a string containing either the input datetime or the current datetime in the given string format

    Arguments:
    dt: (opt) datetime
        The datetime value to convert to a string.
        default: None, in which case the current datetime is used

    format: (opt) str
        Datetime format in the strftime() function to use to generate the datetime string.
        default: "%Y%m%d_%H%M%S" (i.e. in format yyyymmdd_HHMMSS)

    prefix: (opt) str
        Prefix to add to the returned datetime string.
        default: empty string

    suffix: (opt) str
        Suffix to add to the returned datetime string.
        default: empty string

    extension: (opt) str
        Extension to add at the end of the returned datetime string, after any suffix.
        default: empty string

    sep: (opt) str
        Separator between the prefix and the datetime string and between the datetime string and the suffix,
        whenever a prefix/suffix is requested.
        default: empty string
    """
    if dt is None:
        dt_str = datetime.datetime.today().strftime(format)
    else:
        if not isinstance(dt, (datetime.datetime, datetime.date)):
            raise ValueError("Parameter 'dt' must be of type `datetime.datetime` or `datetime.date` (type={})".format(type(dt)))
        dt_str = dt.strftime(format)

    if prefix is not None and prefix != "":
        dt_str = prefix + sep + dt_str
    if suffix is not None and suffix != "":
        dt_str = dt_str + sep + suffix
    dt_str = dt_str + extension

    return dt_str

def is_integer(x):
    "Returns whether the input parameter is an integer (i.e. either int, np.int32, np.int64)"
    return isinstance(x, (int, np.int32, np.int64))

def is_float(x):
    "Returns whether the input parameter is a float number (i.e. either float, np.float32, np.float64)"
    return isinstance(x, (float, np.float32, np.float64))

def is_scalar(x):
    "Returns whether the input parameter is a scalar (i.e. either int, np.int32, np.int64, float, np.float32, np.float64)"
    return is_integer(x) or is_float(x)

def as_array(x):
    "Converts a scalar or list of values to a numpy array"
    if is_scalar(x):
        x = np.array([x])
    else:
        if not isinstance(x, (list, np.ndarray)):
            raise ValueError("Parameter x must be a scalar, list, or numpy array ({})".format(type(x)))
        x = np.array(x)

    return x

def convert_str_argument_to_list_of_type(option, opt, value, parser, type=float):
    "Function to be used as callback by the optparse.OptionParser class when parsing a user argument that should be interpreted as a list of elements of the given `type` (e.g. '[3.0, 2.1]' or '3.0, 2.1')"
    # For debugging, uncomment the following print() lines:
    #print(f"option: {type(option)}, {dir(option)}")
    #print(f"opt: {opt}")
    #print(f"value: {value}")
    #print(f"parser: {parser}")
    #print(f"parser.values: {parser.values}")
    if isinstance(value, str):
        # Set the argument as a list of floats
        setattr(parser.values, option.dest, convert_str_to_list_of_type(value, type=type))

def convert_str_to_list_of_type(str_value, sep="[, ]", type=float):
    """
    A string representing a list (e.g. of numbers) (e.g. '[3.1, 2.5]') is converted to a list of the given type (default `float`).
    This is useful when reading list values from cells in a CSV file.

    It is assumed that the values in the string are convertible to the given number type.
    """
    str_value_as_list = [type(float(s.replace("[", "").replace("]", ""))) for s in re.split(sep, str_value) if len(s) > 0 and s not in ['[', ']']]
        ## NOTES:
        ## - We use type(float()) so that when type=int, the conversion of e.g. '3.48' returns 3 instead of an error!
        ## (the reason is that int('3.48') gives an error whereas float('3.48') does NOT! so we first convert to float and then to integer)
        ## - The `not in` condition is used because when the string is of the form e.g. '[3.48434049 4.9   ]',
        ## then the last element of re.split() is ']' which cannot be converted to a number.
    if len(str_value_as_list) == 1:
        return str_value_as_list[0]
    else:
        return str_value_as_list

def save_objects_to_pickle(object_names, filename, namespace):
    """
    Saves objects into a pickle file stored inside a dictionary whose names are taken from the object_names list and values taken from the given namespace

    Ex: save_objects_to_pickle(['x', 'y', 'z'], "out.pkl", locals())

    Arguments:
    object_names: list of str
        List containing the names of the objects to save from the given namespace.

    filename: str
        Name of the file to which objects are saved in pickle format.

    namespace: dict
        Dictionary containing the object names as keys and the object values as values.
    """
    dict_objects_to_save = dict([(obj_name, namespace[obj_name]) for obj_name in object_names])
    joblib.dump(dict_objects_to_save, filename)

def load_objects_from_pickle(filename, namespace):
    """
    Loads objects saved in a pickle file and stored inside a dictionary whose names are taken from the dictionary keys and values taken from their values

    Arguments:
    filename: str
        Name of the pickle file containing the objects to load.

    namespace: dict
        Dictionary representing the namespace where objects read from the filename should be created, where object names are keys and object values are their values.

    Return: list
    List with the names of the loaded objects into the given namespace.
    """
    try:
        dict_objects = joblib.load(filename)
    except IOError as e:
        print(get_exception_message(e))
    assert isinstance(dict_objects, dict), f"The object saved in the input pickle file must be a dictionary: {dict_objects}"
    object_names = []
    for obj_name, obj_value in dict_objects.items():
        object_names += [obj_name]
        namespace[obj_name] = obj_value

    return object_names

def parse_dict_params(dict_params, dict_params_default):
    """
    Parses a set of user parameters given as a dictionary by
    by matching them with a set of default parameter values,
    completing any missing key with their default value.

    The parsing process accepts nested dictionaries.

    The input dictionary with the user parameters is updated
    by including all keys in the default parameters dictionary
    that are not present in the user parameters dictionary. 
    
    Arguments:
    dict_params: dict
        Dictionary containing user parameter values.

    dict_params_default: dict
        Dictionary containing the default values for parameters.
        Any parameters not included in this dictionary but given in `dict_params` are also taken.

    Example:
    params = {'a': {'w': -7}, 't_new': -13}
    params_default =  {'a': {'w': 8, 'z': {'k': 2, 'y': "ok"}}, 'y': 3}
    parse_dict_params(params, params_default)
    params
    {'a': {'w': -7, 'z': {'k': 2, 'y': "ok"}}, 'y': 3, 't_new': -13}  
    """
    dict_params_keys = dict_params.keys() 
    # Go over all keys in the default params dictionary
    # and recursively retrieve their value when the value is in turn a dictionary
    for key, value in dict_params_default.items():
        if isinstance(value, dict) and key in dict_params_keys:
            parse_dict_params(dict_params[key], dict_params_default[key])
        else:
            # Get the key from the user parameters and if not given, assign its default value 
            dict_params[key] = dict_params.get(key, dict_params_default[key])

def show_exec_params(dict_params):
    "Shows the values of a set of parameters given in a dictionary, in alphabetical order of the keys"
    print("**************** Execution parameters ***********************")
    keys = dict_params.keys()
    keys = sorted(keys)
    for key in keys:
        print("{}: {}".format(key, dict_params[key]))
    print("**************** Execution parameters ***********************")

def set_pandas_options():
    "Sets pandas options to e.g. display all columns and all rows in a data frame"
    pandas_options = dict({'display.width': pd.get_option('display.width'),
                           'display.max_columns': pd.get_option('display.max_columns'),
                           'display.max_rows': pd.get_option('display.max_rows')})

    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    return pandas_options

def reset_pandas_options(pandas_options: dict):
    """Resets pandas options to the ones given in the input dictionary

    Arguments:
    pandas_options: dict
        Dictionary with at least the following entries:
        - 'display.width'
        - 'display.max_columns'
        - 'display.max_rows'
        containing the values of the respective pandas options to reset.
    """
    pd.set_option('display.width', pandas_options['display.width'])
    pd.set_option('display.max_columns', pandas_options['display.max_columns'])
    pd.set_option('display.max_rows', pandas_options['display.max_rows'])

def assert_equal_data_frames(df_observed, df_expected, columns, atol=1E-6, printFlag=False):
    """
    Assert that two data frames are equal up to a given absolute tolerance.
    The function is prepared to compare data frames that may contain lists as some of their cell values and to accept comparison between NaN values.

    df_observed: pandas DataFrame
        Data frame containing the observed data to compare against the expected.

    df_expected: pandas DataFrame
        Data frame containing the expected data against which the observed data is compare.

    columns: list
        List of the column names to compare between the two data frames.
        The columns must exist in both data frames.

    atol: (opt) float
        Absolute tolerance for the comparison between numeric values (used in the call to np.allclose() for each compared value).
        default: 1E-6

    printFlag: bool
        Whether to show which column is compared and their respective observed and expected values for each row.
    """
    # Note: we cannot simply use np.allclose() on all the data frame in the assertion because this fails when some elements of the output data frame are lists
    # (e.g. 'theta_true', etc.) and a comparison of lists fails with np.allclose().
    # The only solution I found was to iterate on the columns and on the rows for each column!
    # Also, using `df_observed == df_expected` instead of np.allclose() does NOT work either
    # because we get the error message that the truth value of a set of values... use all()...
    # And still, if I use all as in `all((df_observed == df_expected).all())` it does not work!!!!!!
    for col in columns:
        if printFlag:
            print(f"col={col}")
        for idx, expected_value in df_expected[col].iteritems():
            observed_value = df_observed[col][idx]
            if printFlag:
                print(f"idx={idx}, observed={observed_value}, expected={expected_value}")
            assert is_scalar(observed_value) and is_scalar(expected_value) or \
                    not is_scalar(observed_value) and len(observed_value) == 1 and is_scalar(expected_value) or \
                    is_scalar(observed_value) and not is_scalar(expected_value) and len(expected_value) == 1 or \
                    len(observed_value) == len(expected_value), "The observed and expected value have the same length"
            assert np.allclose(observed_value, expected_value, atol=atol, equal_nan=True), f"The observed and expected value are the same within an absolute tolerance of {atol}"

def index_linear2multi(idx, shape, order='C'):
    """
    Converts a linear index into a 2D index based on the given shape and order of access as defined in numpy.reshape()"

    Arguments:
    idx: int
        Linear index to convert.

    shape: tuple
        Duple with the 2D dimensions of the array where the converted index will be used.

    order: (opt) char
        The order in which the 2D shape is swept to obtain the linear index.
        Either 'C' for C-like order, i.e. by rows (where the LAST axis index changes fastest back to the FIRST index
        which changes slowest), or 'F' for Fortran-like order, i.e. by columns (where the FIRST axis index changes
        fastest up to the LAST index which changes slowest)
        default: 'C'

    Return: tuple of int
    Duple with the 2D indices to access the element given by the linear index using the given order.
    """
    # TODO: (2021/12/11) Extend this function to multi-index shapes, not only to 2-index shapes (e.g. multi-index of size (3, 2, 4), where len(shape) = 3)
    if len(shape) != 2:
        raise ValueError("The `shape` parameter must be a numpy array of length 2 to indicate a 2D shape ({})".format(shape))

    if idx < 0 or idx >= np.prod(shape):
        raise ValueError("The linear index 'idx={}' is out of bounds ([0, {}])".format(idx, np.prod(shape)-1))

    # Size of each dimension
    n1, n2 = shape

    if order == 'F':
        # Fortran-like order: by columns
        x1, x2 = idx % n1, int( idx / n1 )
    else:
        # C-like order: by rows
        x1, x2 = int( idx / n2 ), idx % n2

    return int(x1), int(x2)

def index_multi2linear(idx_2d, shape, order='C'):
    """
    Converts a 2D index into a linear index based on the given shape and order of access as defined in numpy.reshape()

    Arguments:
    idx: tuple
        2D index to convert.

    shape: tuple
        Duple with the 2D dimensions of the array that can be accessed with the given 2D index.

    order: (opt) char
        The order in which the 2D shape is swept by the linear index.
        Either 'C' for C-like order, i.e. by rows (where the LAST axis index changes fastest back to the FIRST index
        which changes slowest), or 'F' for Fortran-like order, i.e. by columns (where the FIRST axis index changes
        fastest up to the LAST index which changes slowest)
        default: 'C'

    Return: int
    Linear index to access the element of the 2D shape using the given order.
    """
    # TODO: (2021/12/11) Extend this function to multi-index shapes, not only to 2-index shapes (e.g. multi-index of size (3, 2, 4), where len(shape) = 3)
    if len(shape) != 2:
        raise ValueError("The `shape` parameter must be a numpy array of length 2 to indicate a 2D shape ({})".format(shape))

    if idx_2d[0] < 0 or idx_2d[0] >= shape[0]:
        raise ValueError("The first index in 'idx_2d' (x1={}) is out of bounds ([0, {}])".format(idx_2d[0], shape[0]-1))
    if idx_2d[1] < 0 or idx_2d[1] >= shape[1]:
        raise ValueError("The first index in 'idx_2d' (x2={}) is out of bounds ([0, {}])".format(idx_2d[1], shape[1]-1))

    # Size of each dimension
    n1, n2 = shape

    # Positions of all the dimensions
    x1, x2 = idx_2d

    if order == 'F':
        # Fortran-like order: by columns
        idx = x2 * n1 + x1
    else:
        # C-like order: by rows
        idx = x1 * n2 + x2

    return int(idx)


def discretize(x, n, xmin, xmax):
    """    Discretizes a real number into one of n equal-sized left-closed intervals in [xmin, xmax].
    The rightmost interval is right-closed as well, so that xmax is included in the range of possible x values.

    Arguments:
    x: float
        Value to discretize.

    n: int
        Number of equal-sized intervals in which the range [xmin, xmax] is divided into.

    xmin: float
        Minimum value allowed for the input x when discretizing. It's included in the range.

    xmax: float
        Maximum value allowed for the input x when discretizing. It's included in the range.

    Return: int between 0 and n-1
    Value indexing the discretized value of x, which indicates the interval out of the n possible intervals to which x
    belongs. The index ranges 0 to (n-1), where 0 represents the range [xmin, xmin + dx) and n-1 represents the range
    [xmax - dx, xmax] where dx = (xmax - xmin) / n.
    Note that xmax is included on the rightmost interval.
    The index value is obtained by calling np.digitize() which is more robust (and possibly faster) than my method
    in border-line situations (such as when x is exactly equal to the left bound of an interval).
    For implementation details of np.digitize() see:
    https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    where we see that a binary search is carried out in order to find the correct interval.
    """
    # Bound x into [xmin, xmax]
    x_bounded = max(xmin, min(x, xmax))

    # IMPORTANT: np.digitize() returns a value between 1 and the length of bins (second parameter), not between 0 and len-1!!!!!!
    return np.digitize(x_bounded, np.linspace(xmin, xmax, n, endpoint=False), right=False) - 1


def deprecated_discretize(x, n, xmin, xmax):
    """
    WARNING: (2022/06/06) I observed problems of correct discretization due most likely to some precision issues
    (i.e. the calculation done below to find the interval index on which x falls, `int( (x_bounded - xmin) / dx )`
    may not yield the desired result (it happened in the Moutain Car discrete environment with x = -1.02 when -1.02
    was the left bound of the interval with index 1)
    Better use np.digitize() instead: see the new version of discretize() for the implementation.
    For implementation details of np.digitize() see:
    https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    where we see that a binary search is carried out in order to find the correct interval.

    Discretizes a real number into one of n equal-sized left-closed intervals in [xmin, xmax].
    The rightmost interval is right-closed as well, so that xmax is included in the range of possible x values.

    Arguments:
    x: float
        Value to discretize.

    n: int
        Number of equal-sized intervals in which the range [xmin, xmax] is divided into.

    xmin: float
        Minimum value allowed for the input x when discretizing. It's included in the range.

    xmax: float
        Maximum value allowed for the input x when discretizing. It's included in the range.

    Return: int between 0 and n-1
    Value indexing the discretized value of x, which indicates the interval out of the n possible intervals to which x
    belongs. The index ranges 0 to (n-1), where 0 represents the range [xmin, xmin + dx) and n-1 represents the range
    [xmax - dx, xmax] where dx = (xmax - xmin) / n.
    Note that xmax is included on the rightmost interval.
    """
    # Make sure x is float
    x = float(x)

    # Bound x into [xmin, xmax]
    x_bounded = max( xmin, min(x , xmax) )

    # Interval size
    dx = (xmax - xmin) / n

    # Discrete value (we consider the special case when x_bounded == xmax so that the value belongs to the rightmost interval)
    x_discrete = (n - 1) if x_bounded == xmax else int( (x_bounded - xmin) / dx )

    assert isinstance(x_discrete, int), "The discretized value is an integer"
    assert 0 <= x_discrete < n, "The discretized value is between 0 and n-1={} ({})".format(n-1, x_discrete)

    return x_discrete


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
        if is_scalar(size) or len(size) == 1:
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
    # returns an index number that is 1 + the index where
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

def find_signed_max_value(x):
    """
    Finds the maximum value in an array or list based on their absolute values and returns the maximum value found with its original sign

    Arguments
    x: array, list or tuple
        Values to analyze

    Return: float
    The maximum absolute value with its original sign, i.e. before taking the absolute value.
    """
    # e.g. x = [4, -5, 0, -3]
    x = np.array(x)

    ind_nonneg = (x >= 0)
    ind_neg = [not ind for ind in ind_nonneg]

    if sum(ind_nonneg) > 0:
        x_max_pos = np.max(x[ind_nonneg])  # e.g. +4
    else:
        x_max_pos = 0.0

    if sum(ind_neg) > 0:
        x_max_neg = np.min(x[ind_neg])  # e.g. -5
    else:
        x_max_neg = 0.0

    max_signed_value = x_max_neg if np.abs(x_max_neg) > np.abs(x_max_pos) else x_max_pos  # e.g. -5)

    return max_signed_value

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

    Return: int
        The smallest index in `alist` that is equal to `value` or -1 if `value` is not present in `alist`.
    """
    for i, v in enumerate(alist):
        if v == value:
            return i
    return -1

def find_last(alist, value):
    """
    Returns the largest index in `alist` that is equal to `value`

    Arguments:
    alist: list
        List where `value` is searched for.
    value: any object
        A value/object to search in `alist`.

    Return: int
        The largest index in `alist` where `value` is found  or -1 if `value` is not found.
    """
    for i in range(len(alist)-1, -1, -1):
        if alist[i] == value:
            return i
    return -1

def find_first_value_in_list(alist, value):
    """
    Returns the smallest index in `alist` (a list of lists) that contains `value`

    Arguments:
    alist: list of lists
        List in whose elements, which are lists, `value` is searched for.
    value: any object
        A value/object to search in `alist` as element of one of its lists.

    Return: int
        The smallest index in `alist` where `value` is found  or -1 if `value` is not found.
    """
    for i, values in enumerate(alist):
        if value in values:
            return i
    return -1

def find_last_value_in_list(alist, value):
    """
    Returns the largest index in `alist` (a list of lists) that contains `value`

    Arguments:
    alist: list of lists
        List in whose elements, which are lists, `value` is searched for.
    value: any object
        A value/object to search in `alist` as element of one of its lists.

    Return: int
        The largest index in `alist` where `value` is found  or -1 if `value` is not found.
    """
    for i in range(len(alist)-1, -1, -1):
        if value in alist[i]:
            return i
    return -1

# DM-2024/03/04: NOTE: there is also the np.searchsorted() function which tells us at what index a new element would be inserted in a sorted array, so essentially it does the same thing as this function...
def find_smaller_and_closest_in_sorted_array(arr, value, check=False):
    """
    Finds the index in a SORTED array (in ascending order) whose value is the closest to the given value but smaller or equal

    The check of the sort condition is ONLY done when `check=True`.

    Arguments:
    arr: numpy array
        Sorted array, in ascending order, of values where the searched-for `value` is performed.

    value: float
        Value that is searched for in `arr`.

    check: (opt) bool
        Whether to check whether the array is sorted in ascending order.
        default: False

    Return: int
    A value between -1 and len(arr) - 1 containing the index in `arr` satisfying the condition described above in relation to `value`.
    If the returned value is -1, it means that `value` is smaller than all values in the array.
    """
    if check:
        if not all(arr == sorted(arr)):
            raise ValueError(f"The input array must be sorted in ascending sequence:\n{arr}")

    # Extend the input array with a +Inf value at the end so that we can easily take care of the case where `value` is larger than all array values,
    # in which case the index of the last element in the array should be returned.
    arr = np.r_[arr, +np.Inf]
    return np.argmax(value < arr) - 1

def list_contains_either(container :list, content):
    "Checks whether a list contains an element or at least one element of another list"
    if not isinstance(content, list):
        content = [ content ]

    if container is not None:
        for elem in container:
            if elem in content:
                return True

    return False

def merge_values_in_time(t1, y1, t2, y2, unique=False):
    """
    Merges two lists of values measured at different times into two separate lists of values measured both at the same times

    When a time value does not appear in one of the time lists being merged, its y value is kept constantly equal to the y value observed
    at the time value observed latest in the list.

    When `unique=False` and duplicate time values (> 0) appear in any of the time lists, the duplicate time values appear as many times as the sum of
    the duplicate count in both lists, extending each list with the y value corresponding to the last duplicate time value.
    Ex:
    t1 = [0, 2.5, 2.5, 2.5, 3.2]
    y1 = [1    2,   3,   4,  10]
    t2 = [0, 2.5, 2.5, 4.9]
    y2 = [-1, -2,  -3,  -4]
    then the result of the merge when unique=False is:
    t_merged  = [0, 2.5, 2.5, 2.5, 2.5, 2.5, 3.2, 4.9]  # the repeated time value 2.5 appears (3 + 2) times because 2.5 appears 3 times in the first time list and 2 times in the second time list.
    y1_merged = [1,   2,   3,   4,   4,   4, 10,   10]  # `4` is the value taken by y1 in the LAST duplicate record with t1=2.5
    y2_merged = [-1, -2,  -3,  -3,  -3,  -3, -3,   -4]  # `-3` is the value taken by y2 in the LAST duplicate record with t2=2.5

    ASSUMPTIONS:
    - The length of each time list (ti) is the same as the list of respective measurements (yi).
    - Time value 0 MUST be present at the FIRST element of each time list and must appear only once.
    - The two time lists are sorted increasingly.

    Arguments:
    t1: list
        List of times associated to the first list of measurements y1, assumed sorted increasingly. 

    y1: list
        List of measurements associated to the first times list t1.

    t2: list
        List of times associated to the second list of measurements y2, assumed sorted increasingly. 

    y2: list
        List of measurements associated to the second times list t2.

    unique: (opt) bool
        Whether the merged t values should be unique when merging values in input t1 and t2.
        If repeated values are found in the input time lists t1 and t2, the respective y1 or y2 value
        associated to the index of the LAST occurrence of the repeated time value is:
        - kept, when unique=True
        - repeated further to the right of the current time value, when unique=False
        default: False

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

    # Note: t1_merged, t2_merged, y1_merged, y2_merged will be updated below when doing the actual merge
    if unique:
        # Remove duplicates from each input time list to merge
        # For each measurement series, the y value for the LAST occurrence of the repeated t value is kept
        # Note that we add a fictitious +Inf value for both the list of times (t1, t2) and the respective list of values (y1, y2) in order to take into account the case
        # when there are duplicate t values at the END of the list. Otherwise, the duplicates are not removed by keeping the last duplicate record, but by keeping the
        # previous to last duplicate record.

        # Process first list to merge
        t1_merged, idx1 = np.unique(t1 + [np.Inf], return_index=True)
        t1_merged = list(t1_merged)[:-1]                            # `:-1`: remove the last time corresponding to the fictitious +np.Inf
        idx1_next = list(idx1[1:]) + list([ idx1[-1]+1 ])           # Compute the indices of the start of the next duplicate groups in order to keep the LAST duplicate record in each group by keeping the idx1_next-1 index in the y list
        y1_merged = [(y1 + [np.Inf])[i-1] for i in idx1_next][:-1]  # `:-1` remove the last y value corresponding to the fictitious +np.Inf

        # Process second list to merge
        t2_merged, idx2 = list( np.unique(t2 + [np.Inf], return_index=True) )
        t2_merged = list(t2_merged)[:-1]                            # Ditto t1_merged
        idx2_next = list(idx2[1:]) + list([ idx2[-1]+1 ])           # Ditto idx1
        y2_merged = [(y2 + [np.Inf])[i-1] for i in idx2_next][:-1]  # Ditto y1_merged
    else:
        t1_merged = t1.copy()
        y1_merged = y1.copy()
        t2_merged = t2.copy()
        y2_merged = y2.copy()

    # Insert values in the first pair of lists
    ntimes_t0 = 0   # Counter to check that the number of time values equal to 0 is only one
    for t in t2:
        if t == 0:
            ntimes_t0 += 1
            if ntimes_t0 > 1:
                raise ValueError(f"Input time series t1 has at most one value equal to 0, {ntimes_t0} were found.")
        else:
            idx_insort, found = insort(t1_merged, t, unique=unique)
            assert idx_insort > 0, "The index to insert the new time must be larger than 0 ({})".format(idx_insort)
            if not unique or not found:
                y1_merged.insert(idx_insort, y1_merged[idx_insort-1])

    # Insert values in the second pair of lists
    ntimes_t0 = 0   # Counter to check that the number of time values equal to 0 is only one
    for t in t1:
        if t == 0:
            ntimes_t0 += 1
            if ntimes_t0 > 1:
                raise ValueError(f"Input time series t1 has at most one value equal to 0, {ntimes_t0} were found.")
        else:
            idx_insort, found = insort(t2_merged, t, unique=unique)
            assert idx_insort > 0, "The index to insert the new time must be larger than 0 ({})".format(idx_insort)
            if not unique or not found:
                y2_merged.insert(idx_insort, y2_merged[idx_insort-1])

    assert len(t1_merged) == len(y1_merged) == len(t2_merged) == len(y2_merged), \
            "The length of the first pair of merged lists is the same" \
            " as that of the second pair of merged lists ({}, {}, {}, {})" \
            .format(len(t1_merged), len(y1_merged), len(t2_merged), len(y2_merged))
    #assert t1_merged == t2_merged, \
    #        "The times lists resulting from the merge are the same ({}, {})" \
    #        .format(t1_merged, t2_merged)

    return t1_merged, y1_merged, y2_merged

def aggregation_bygroups(df, groupvars, analvars,
                         stats=["count", "mean", "std", "min", "max"]):
                        ## NOTE: (2023/03/12) To understand why PyCharm gives the warning that the default value of `stat` is mutable, see this link: http://www.omahapython.org/IdiomaticPython.html#default-parameter-values
    """
    Computes the given summary statistics in the input data frame on the analysis variables
    by the given group variables.
    
    Arguments:
    groupvars: str or list
        String with the grouping variable name or list of grouping variables.

    analvars: str or list
        String with the analysis variable name or list of analysis variables
        whose statistics is of interest.

    stats: list
        List with the summary statistics names and/or functions to compute them.
        When both "std" and "n" are requested the standard error is also computed
        as "std" / sqrt("n') and stored in the output dataset as statistic "SE".
        default: ["count", "mean", "std", "min", "max"]

    Return: data frame
    Data frame containing the summary statistics results.
    If the 'std' and 'n' (count) summary statistics is part of the `dict_stats` dictionary,
    the Standard Error (SE) is also computed as 'std' / sqrt('n')
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Parameter 'df' must be a pandas DataFrame ({})".format(type(df)))

    # Cast all analysis variables to float in order to compute statistics like mean, std, etc.!!
    # Otherwise the statistics computed on integer variables are also integer-valued!
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
    for var in analvars:
        df = df.astype({var: np.float})

    # Create the groupby object
    df_grouped = df.groupby(groupvars, as_index=True, group_keys=False)
        ## I don't know what's the effect of group_keys=False as I don't see any difference with group_keys=True

    # Aggregate the analysis variables
    df_agg = df_grouped[analvars].agg(stats)

    # Compute the Standard Error (SE) when both 'std' and 'n' have been requested
    if 'std' in stats and 'count' in stats:
        df_agg_se = df_agg.xs('std', axis=1, level=1) / np.sqrt( df_agg.xs('count', axis=1, level=1) )
        df_agg_se.set_axis([analvars, ['SE']*len(analvars)], axis=1, inplace=True)
        # Concatenate the columns and reorder them with reindex() so that each variable's statistics appear together!
        # Ref: https://stackoverflow.com/questions/11194610/how-can-i-reorder-multi-indexed-dataframe-columns-at-a-specific-level
        df_agg = pd.concat([df_agg, df_agg_se], axis=1).reindex(analvars, axis=1, level=0)

    return df_agg


if __name__ == "__main__":
    #---------------------- parse_dict_params -------------------------#
    print("\n--- parse_dict_params() ---")
    params = {'a': {'w': -7}, 't_new': -13, 't_new_dict': {'a': 2, 'b': {'c': 5, 'd': None}}}
    params_default =  {'a': {'w': 8, 'z': {'k': 2, 'y': "ok"}},    # nested parameters
                       'x': {'a': 2, 'f': 5},   # 'a' key is repeated in this nested dictionary
                       'y': 3}
    parse_dict_params(params, params_default)
    print(params)
    # NOTE: The order of the keys in both compared dictionaries don't have any influence in the comparison, GREAT! :)
    assert params == {'a': {'w': -7, 'z': {'k': 2, 'y': "ok"}},
                      'x': {'a': 2, 'f': 5}, 'y': 3, 
                      't_new': -13, 't_new_dict': {'a': 2, 'b': {'c': 5, 'd': None}}}
    #---------------------- parse_dict_params -------------------------#


    #-------- index_multi2linear / index_linear2multi -----------------#
    print("\n--- index_multi2linear() and index_linear2multi() ---")
    # 2D matrix with linear indices as its values
    M = np.arange(12).reshape([4, 3])
    assert index_multi2linear([2, 1], M.shape) == 7                 # This is the linear index in the by-column numbering of the 2D-cells in matrix M
    assert index_linear2multi(7, M.shape) == (2, 1)
    assert index_linear2multi(6, M.shape) == (2, 0)

    assert index_multi2linear([2, 1], M.shape, order='F') == 6      # This is the linear index in the by-column numbering of the 2D-cells in matrix M
    assert index_linear2multi(6, M.shape, order='F') == (2, 1)
    assert index_linear2multi(7, M.shape, order='F') == (3, 1)

    # Out of range indices
    # Note that in order to use unittest.TestCase.assertRaises() we need to create an INSTANCE of the TestCase class
    # Otherwise we get the error "assertRaises() arg 1 must be an exception type or tuple of exception types"
    # Ref: https://stackoverflow.com/questions/18084476/is-there-a-way-to-use-python-unit-test-assertions-outside-of-a-testcase
    # which was referenced by Martijn Pieters in https://stackoverflow.com/questions/49369163/custom-exceptions-in-unittests
    tc = TestCase()
    tc.assertRaises(ValueError, index_multi2linear, [4, 0], M.shape)
    tc.assertRaises(ValueError, index_multi2linear, [0, 3], M.shape)
    tc.assertRaises(ValueError, index_multi2linear, [-1, 2], M.shape)
    tc.assertRaises(ValueError, index_multi2linear, [0, -1], M.shape)
    tc.assertRaises(ValueError, index_linear2multi, 12, M.shape)
    tc.assertRaises(ValueError, index_linear2multi, -1, M.shape)
    #-------- index_multi2linear / index_linear2multi -----------------#


    #--------------------------- discretize ---------------------------#
    print("\n--- discretize() ---")
    n = 5; xmin = 3.2; xmax = 7.8                    # This gives the following intervals: [3.2 , 4.12, 5.04, 5.96, 6.88] (obtained with np.linspace(3.2, 7.8, 5, endpoint=False) --note `endpoint=False`)
    assert discretize(5.6, n, xmin, xmax) == 2       # Any number
    assert discretize(xmin, n, xmin, xmax) == 0      # Extremes
    assert discretize(xmax, n, xmin, xmax) == n-1    # Extremes
    assert discretize(xmin - 0.16, n, xmin, xmax) == 0      # Out of bounds
    assert discretize(xmax + 0.23, n, xmin, xmax) == n-1    # Out of bounds
    assert discretize(4.12, n, xmin, xmax) == 1       # Left bound of an interval
    assert discretize(4.11999999, n, xmin, xmax) == 0 # Close to right bound of an interval

    # Negative min, positive max
    n = 5; xmin = -5.1; xmax = 7.8                   # This gives the following intervals: [-5.1, -2.52, 0.06, 2.64, 5.22] (obtained with np.linspace(-5.1, 7.8, 5, endpoint=False) --note `endpoint=False`)
    assert discretize(5.6, n, xmin, xmax) == 4       # Any number
    assert discretize(0.0, n, xmin, xmax) == 1       # 0.0
    assert discretize(xmin, n, xmin, xmax) == 0      # Extremes
    assert discretize(xmax, n, xmin, xmax) == n-1    # Extremes
    assert discretize(xmin - 0.16, n, xmin, xmax) == 0      # Out of bounds
    assert discretize(xmax + 0.23, n, xmin, xmax) == n-1    # Out of bounds
    assert discretize(-2.52, n, xmin, xmax) == 1            # Left bound of an interval
    assert discretize(-2.5200000001, n, xmin, xmax) == 0    # Close to right bound of an interval
    #--------------------------- discretize ---------------------------#


    #----------------------- find_signed_max_value --------------------#
    print("\n--- find_signed_max_value() ---")
    # Signed max value is negative
    x = [4, -5, 0, -3]
    assert find_signed_max_value(x) == -5

    # Signed max value is positive and trying an array
    x = np.array([4, -5.0, 0, -3, 6.1])
    assert find_signed_max_value(x) == +6.1

    # Tied negative max and positive max (returns positive)
    x = [5.0, -5.0, 0, -3]
    assert find_signed_max_value(x) == +5

    # All negative values
    x = [-4.8, -5, -0.5, -3]
    assert find_signed_max_value(x) == -5

    # All positive values
    x = [4, 5, 0.3, 3]
    assert find_signed_max_value(x) == 5

    # No zeros
    x = [4, -5, 0.3, 3]
    assert find_signed_max_value(x) == -5
    # ...as array
    x = np.array([4, -5, 0.3, 3])
    assert find_signed_max_value(x) == -5

    # All zeros
    x = [0, 0, 0]
    assert find_signed_max_value(x) == 0
    #----------------------- find_signed_max_value --------------------#

    
    #----------------- find_first/last_value_in_list ------------------#
    print("\n--- find_first/last_value_in_list() ---")
    ll = [[1, 3], ['A', 'B'], [3]]
    assert find_first_value_in_list(ll, 3) == 0
    assert find_first_value_in_list(ll, 'B') == 1
    assert find_first_value_in_list(ll, 'C') == -1

    assert find_last_value_in_list(ll, 3) == 2
    assert find_last_value_in_list(ll, 'B') == 1
    assert find_last_value_in_list(ll, 'C') == -1
    #----------------- find_first/last_value_in_list ------------------#


    #----------------- find_smaller_and_closest_in_sorted_array --------------#
    # The input array must be sorted increasingly!
    arr = np.array([2, 5, 7])
    # Find the closest value to a value NOT present in the array
    assert find_smaller_and_closest_in_sorted_array(arr, 2.5) == 0
    assert find_smaller_and_closest_in_sorted_array(arr, 5.3) == 1
    assert find_smaller_and_closest_in_sorted_array(arr, 7.2) == 2
    # Find the closest value to a value NOT present in the array, one test per array element
    assert find_smaller_and_closest_in_sorted_array(arr, 2) == 0
    assert find_smaller_and_closest_in_sorted_array(arr, 5) == 1
    assert find_smaller_and_closest_in_sorted_array(arr, 7) == 2
    # Check sort condition and check ValueError exception when the condition is not satisfied
    import pytest
    with pytest.raises(ValueError):
        find_smaller_and_closest_in_sorted_array(arr[::-1], 2.5, check=True)
    #----------------- find_smaller_and_closest_in_sorted_array --------------#


    #---------------------- list_contains_either  ---------------------#
    print("\n--- list_contains_either() ---")
    container = [1, 3, 3]
    content1 = 3
    content2 = 2
    content3 = [2, 3]
    content4 = [1, 2]
    content5 = [5, 5, 7]
    assert list_contains_either(container, content1)
    assert not list_contains_either(container, content2)
    assert list_contains_either(container, content3)
    assert list_contains_either(container, content4)
    assert not list_contains_either(container, content5)
    #---------------------- list_contains_either  ---------------------#
    
    
    #-------------------- merge_values_in_time ------------------------#
    print("\n--- merge_values_in_time(): Test #1 on unique time values across series of different lengths")
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
    print("\n--- merge_values_in_time(): Test #2 on UNIQUE time values WITHIN series but some common time values across series (UNIQUE=True)")
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
    print("\n--- merge_values_in_time(): Test #3 on UNIQUE time values WITHIN series but some common time values across series (UNIQUE=False)")
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
    print("\n--- merge_values_in_time(): Test #4 on REPEATED time values in the MIDDLE of each series and a few common time values across series (UNIQUE=True)")
    t1 = [0.0, 2.5, 2.5, 2.5, 3.2,   3.2, 3.2, 7.2, 11.3]
    y1 = [  4,   3,   2,   1,   4,     5,   6,   5,    4]
    t2 = [0.0, 1.1, 2.5, 2.5, 2.87, 3.3, 4.8,  6.9, 7.5]
    y2 = [  0,   0,   1,   3,    2,   1,   1,    0,   2]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=True)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 7.5, 11.3]
    assert y1f == [4,   4,   1,    1,   6,   6,   6,   6,   5,   5,    4]
    assert y2f == [0,   0,   3,    2,   2,   1,   1,   0,   0,   2,    2]

    print("\n--- merge_values_in_time(): Test #5 same as previous test but with Unique=False")
    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=False)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.87, 3.2, 3.2, 3.2, 3.3, 4.8, 6.9, 7.2, 7.5, 11.3]
    assert y1f == [4,   4,   3,   2,   1,   1,   1,    1,   4,   5,   6,   6,   6,   6,   5,   5,    4]
    assert y2f == [0,   0,   1,   3,   3,   3,   3,    2,   2,   2,   2,   1,   1,   0,   0,   2,    2]

    #-------------------------
    print("\n--- merge_values_in_time(): Test #6 on REPEATED time values at the END of each series and a few common time values across series (UNIQUE=True)")
    t1 = [0.0, 2.5, 3.2, 3.2, 3.2, 7.2, 11.3, 11.3]
    y1 = [  4,   3,   4,   5,   6,   5,    4,  0.5]
    t2 = [0.0, 1.1, 2.50, 2.87, 3.3, 4.8, 6.9, 20.0]
    y2 = [  0,   0,    1,    2,   1,   1,   0,    3]

    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=True)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.87, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3, 20.0]
    assert y1f == [4,   4,   3,    3,   6,   6,   6,   6,   5,  0.5,  0.5]
    assert y2f == [0,   0,   1,    2,   2,   1,   1,   0,   0,    0,    3]

    print("\n--- merge_values_in_time(): Test #7 same as previous test but with Unique=False")
    t, y1f, y2f = merge_values_in_time(t1, y1, t2, y2, unique=False)

    print("Merged lists:")
    print(np.c_[t, y1f, y2f])
    assert t == [0.0, 1.1, 2.5, 2.5, 2.87, 3.2, 3.2, 3.2, 3.3, 4.8, 6.9, 7.2, 11.3, 11.3, 20.0]
    assert y1f == [4,   4,   3,   3,    3,   4,   5,   6,   6,   6,   6,   5,    4,  0.5,  0.5]
    assert y2f == [0,   0,   1,   1,    2,   2,   2,   2,   1,   1,   0,   0,    0,    0,    3]
    #-------------------- merge_values_in_time ------------------------#


    #------------------ convert_str_to_list_of_type -------------------#
    print("\n--- convert_str_to_list_of_type()")
    # "Normal" case
    assert np.allclose(convert_str_to_list_of_type('[3.48, -4.9]'), [3.48, -4.9]) # Separated by comma
    assert np.allclose(convert_str_to_list_of_type('[3.48 -4.9]'), [3.48, -4.9])  # Separated by space
    # Mix of comma and spaces as separator, of which there can be several, even several commas (which do not separate anything)
    assert np.allclose(convert_str_to_list_of_type('[3.48    -4.9     , , 34.2]'), [3.48, -4.9, 34.2])

    # Convert to integer
    assert np.allclose(convert_str_to_list_of_type('[3    4     , , -34]', type=int), [3, 4, -34])
    assert np.allclose(convert_str_to_list_of_type('[3.48    4.9     , , -34]', type=int), [3, 4, -34])
    #------------------ convert_str_to_list_of_type -------------------#
