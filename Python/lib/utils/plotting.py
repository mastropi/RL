# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:07:56 2020

@author: Daniel Mastropietro
@description: Functions for customized plotting.
"""

from warnings import warn
import copy

import numpy as np
import matplotlib.pyplot as plt

from .basic import parse_dict_params, aggregation_bygroups


def default_plot_options():
    return {'axis': {'fontsize': 13,
                     'limits': {'xmin': None, 'xmax': None,
                                'ymin': 0, 'ymax': None,
                                'allequal': True},
                     'remove_yaxis': False,
                     'tickmarks': {'x2': None, 'y2': None}
                     },
            'labels':   {'x': None, 'y': None, 'x2': None, 'y2': None},
            'multipliers':  {'x': 1, 'y': 1, 'error': 2},
            'properties':   {
                             'color': "black", 'marker': '.', 'markersize': 5, 'linestyle': 'None',
                             'color_center': "blue", 'marker_center': "x", 'linestyle_center': "solid",
                             'capsize': 4
                             },
            'showtitle':     True,
            'stats':        {'center': "mean", 'scale': "std"},
            'texts':        {'title': None}
            }    

def deprecated_errorbars_standalone(df, x, y, yref=None,
                         figsize=(4,4),    #(8,4)    #(width, height)
                         subplots=(1,1),   #(1,2)    #(height, width) GRRRRRRRRRRRRRRRRRRR!!@!@#@@#@#$@#!!! 
                         # Plot options with keys sorted alphabetically
                         dict_options=dict()): # The default value is given below, so that we do NOT repeat their values here and below
    """
    Generates one or more subplots of repeated measures of variable `y` for each value of `x`
    including error bars associated with a measure of center and scale of the observed `y` values
    for each `x`.  

    Arguments:
    df: pandas.DataFrame
        Dataframe containing the columns to plot.

    x: str or list
        String or list with the column names in `df` to plot on the X-axis of each subplot.
        If a list, the number of elements should coincide with the number of elements in `y`
        and the number of subplots derived from `subplots`.

    y: str or list
        String or list with the column names in `df` to plot on the Y-axis of each subplot.
        If a list, the number of elements should coincide with the number of elements in `x`
        and the number of subplots derived from `subplots`.

    yref: (opt) str
        Name of the column in `df` containing a reference Y-value to plot.
        The value is taken from the first observation in `df`.

    figsize: (opt) list or tuple
        Size of the figure to plot: width x height.
        default: (4,4)

    subplots: (opt) list or tuple
        Definition of subplots to include in the figure: height x width.
        default: (1,1)

    dict_options: (opt) dict
        Dictionary of plotting options.
        default: {  'axis': {   'fontsize': 9,
                                'limits': {'xmin': None, 'xmax': None,
                                           'ymin': 0, 'ymax': None,
                                           'allequal': True},
                                # Should the Y-axis be removed when NOT the leftmost Y-axis?
                                # (only makes sense when the 'allequal' attribute above is True,
                                # i.e. when all axis have the same scale)
                                'remove_yaxis': False
                            },
                    'labels':   {'x': None, 'y': None},  
                                    ## If different x and y variables are to be plotted,
                                    ## labels should be given as lists having the same length
                                    ## as the number of subplots to make,
                                    ## numbered sequentially along the rows first.
                    'multipliers':  {'x': 1, 'y': 1, 'error': 2},
                    'properties':   {# Properties for points for each subplot
                                    'color': ["black"], 'marker': ['.'], 'markersize': [5], 'linestyle': ['None'],
                                    # Properties for error bars and centers for each subplot
                                    'color_center': ["blue"], 'marker_center': ["x"],
                                    'capsize': [4]
                                    },
                    'showtitle':     True,
                    'stats':        {'center': "mean", 'scale': "std"},
                    'texts':        {'title': None, 'legend': None}
                }
    """

    def setup_properties(dict_properties, nplots):
        "Make sure there is one property value per plot when only one given"
        for key, value in dict_properties.items():
            if not isinstance(value, list):
                # Convert the value to a list of length 1 (e.g. ["red", "red"] for property 'color')
                dict_properties[key] = [value]*nplots
            elif len(value) != nplots:
                raise ValueError("The number of plot property values for key {} in the properties dictionary (value={}) must match the number of plots to produce ({})" \
                                 .format(key, value, nplots))

    #----- Parse input parameters -----
    # Parse options
    dict_options_default = {'axis': {'fontsize': 9,
                                     'limits': {'xmin': None, 'xmax': None,
                                                'ymin': 0, 'ymax': None,
                                                'allequal': True},
                                     'remove_yaxis': False
                                     },
                            'labels':   {'x': None, 'y': None},
                            'multipliers':  {'x': 1, 'y': 1, 'error': 2},
                            'properties':   {
                                         'color': "black", 'marker': '.', 'markersize': 5, 'linestyle': 'None',
                                         'color_center': "blue", 'marker_center': "x",
                                         'capsize': 4
                                         },
                            'showtitle':     True,
                            'stats':        {'center': "mean", 'scale': "std"},
                            'texts':        {'title': None, 'legend': None}
                            }
    parse_dict_params(dict_options, dict_options_default)

    (nrows, ncols) = (subplots[0], subplots[1])
    n_subplots = nrows * ncols
    setup_properties(dict_options['properties'], n_subplots)

    # Convert any variable that is a string to a LIST
    if isinstance(x, str) and isinstance(y, str):
        warn("More than one subplot was requested but all 'x' and 'y' variables to plot are the same.")
    if isinstance(x, str):
        x = [x]*n_subplots
    if isinstance(y, str):
        y = [y]*n_subplots
    if isinstance(dict_options['labels']['x'], str):
        dict_options['labels']['x'] = [ dict_options['labels']['x'] ] * n_subplots 
    if isinstance(dict_options['labels']['y'], str):
        dict_options['labels']['y'] = [ dict_options['labels']['y'] ] * n_subplots

    if n_subplots > 1 and dict_options['axis']['remove_yaxis'] and not dict_options['axis']['limits']['allequal']:
        warn("The non-leftmost Y-axis will be removed from the plots but the Y-axis may be different!", \
             "\nSet dict_options['axis']['limits']['allequal']=True if you want to remove Y-axis from non-leftmost plots.")

    # Assertions
    assert isinstance(x, list)
    assert isinstance(y, list)
    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == n_subplots
    assert len(y) == n_subplots
    if dict_options['texts']['legend'] is not None:
        if yref is None:
            assert len(dict_options['texts']['legend']) == 2, \
                "The legend should be a list with 2 labels: points, and error bars. ({})" \
                .format(dict_options['texts']['legend'])
        else:
            assert len(dict_options['texts']['legend']) == 3, \
                "The legend should be a list with 3 labels: points, error bars, and true value." \
                .format(dict_options['texts']['legend'])
    #----- Parse input parameters -----

    # Create the axes on which the plots will be produced
    axes = plt.figure(figsize=figsize).subplots(nrows=subplots[0], ncols=subplots[1])
    if n_subplots == 1:
        # Convert axes to an array, so that everything works fine from now on
        axes = np.array([axes])
    # Convert the axes array to a 1D array so that we can easily go over all of them
    axes1D = axes.reshape((n_subplots,), order='C')    # 'C' is the default, i.e. "C language"-order (along the rows first), as opposed to Fortran order (along the columns first)

    for k, ax in enumerate(axes1D):
        #-- Raw values
        points = ax.plot(df[ x[k] ] * dict_options['multipliers']['x'],
                         df[ y[k] ] * dict_options['multipliers']['y'],
                         color=dict_options['properties']['color'][k],
                         marker=dict_options['properties']['marker'][k], markersize=dict_options['properties']['markersize'][k],
                         linestyle=dict_options['properties']['linestyle'][k])

        #-- Error bars
        # Aggregate by x so that we can compute the center and error
        df_groupby_x = aggregation_bygroups(df, [x[k]], [y[k]],
                                            stats=['count', dict_options['stats']['center'], dict_options['stats']['scale']])

        errors = ax.errorbar(list(df_groupby_x.index), df_groupby_x[ y[k] ][ dict_options['stats']['center'] ] * dict_options['multipliers']['y'],
                           yerr=dict_options['multipliers']['error'] * df_groupby_x[ y[k] ][ dict_options['stats']['scale'] ] * dict_options['multipliers']['y'],
                           capsize=dict_options['properties']['capsize'][k],
                           color=dict_options['properties']['color_center'][k], marker=dict_options['properties']['marker_center'][k])
        legend_objects = [points[0], errors]
        if yref is not None:
            line_ref = ax.hlines(df.iloc[0][yref] * dict_options['multipliers']['y'], df.iloc[0][ x[k] ], df.iloc[-1][ x[k] ], color='gray', linestyles='dashed')
            legend_objects += [line_ref]

        #-- Legend
        if dict_options['texts']['legend'] is None:
            legend_texts = [y[k] + (dict_options['multipliers']['y'] == 1 and " " or "*{}".format(dict_options['multipliers']['y'])),
                           "Avg(" + y[k] + ") +/- " + (dict_options['multipliers']['error'] == 1 and " " or str(dict_options['multipliers']['error'])) + "SE"]
            if yref is not None:
                legend_texts += ["Reference value"]
        else:
            legend_texts = dict_options['texts']['legend']
        ax.legend(legend_objects, legend_texts)#, fontsize='x-small')

    #-- Axis limits
    if dict_options['axis']['limits']['allequal']:
        xmin = np.Inf
        xmax = -np.Inf
        ymin = np.Inf
        ymax = -np.Inf
        for ax in axes:
            # Axis limits
            xmin = min(xmin, ax.get_xlim()[0]); xmax = max(xmax, ax.get_xlim()[1])
            ymin = min(ymin, ax.get_ylim()[0]); ymax = max(ymax, ax.get_ylim()[1])
    else:
        xmin = xmax = ymin = ymax = None

    # Update any of the axis limits if they are given by the user
    # and set the axis labels, if any given
    xmin = dict_options['axis']['limits'].get('xmin', xmin)
    xmax = dict_options['axis']['limits'].get('xmax', xmax)
    ymin = dict_options['axis']['limits'].get('ymin', ymin)
    ymax = dict_options['axis']['limits'].get('ymax', ymax)
    for k, ax in enumerate(axes1D):
        if xmin is None:
            xmin = ax.get_xlim()[0]
        if xmax is None:
            xmax = ax.get_xlim()[1]
        if ymin is None:
            ymin = ax.get_ylim()[0]
        if ymax is None:
            ymax = ax.get_ylim()[1]
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Axis labels
        if dict_options['labels']['x'] is not None:
            ax.set_xlabel(dict_options['labels']['x'][k], fontsize=dict_options['axis']['fontsize'])
        if dict_options['labels']['y'] is not None:
            ax.set_ylabel(dict_options['labels']['y'][k], fontsize=dict_options['axis']['fontsize'])

        # Remove ticks and labels from the right plot as the axis is the same as on the left plot
        if dict_options['axis']['remove_yaxis']:
            if np.mod(k, ncols) > 0:
                # This is not the leftmost y axis, so remove it         
                ax.yaxis.set_ticks([]); ax.yaxis.set_ticklabels([])

    if dict_options['showtitle'] and dict_options['texts']['title'] is not None:
        plt.suptitle(dict_options['texts']['title'])

    return axes

def plot(plotting_func,
         df, xvars, yvars,
         figsize=(4,4),    #(8,4)    #(width, height)
         subplots=(1,1),   #(1,2)    #(height, width) GRRRRRRRRRRRRRRRRRRR!!@!@#@@#@#$@#!!!
         yref=None, yref_legend="Reference value",
         # Plot options with keys sorted alphabetically
         dict_options=dict()): # The default value is given below, so that we do NOT repeat their values here and below
    """
    Calls a plotting function on the specified data frame and variables
    by creating as many plots as needed in the specified subplot layout
    and using the given plotting options.

    Arguments:
    plotting_func: function
        Function to call for the generation of each subplot.
        This function should receive the following arguments:
        - ax: AxesSubplot object of the matplotlib.axes._subplots module where the plot should be generated;
                i.e. this is an instance of a subplot returned by the matplotlib.pyplot.figure().subplots() method.
        - df: Pandas Dataframe containing the data to plot.
        - x: Name of the column in `df` to use on the X-axis of the plot.
        - y: Name of the column in `df` to use on the Y-axis of the plot.
        - dict_options: dictionary containing plotting properties, such as markers, colors, labels, etc.
        We use the convention that functions that are accepted as argument here
        are named `plot_*()` e.g. plot_errorbars().

    df: pandas.DataFrame
        Data to plot.

    xvars: string or list
        Name(s) of the columns in `df` containing the variables to plot in the X-axis of each subplot.

    yvars: string or list
        Name(s) of the columns in `df` containing the variables to plot in the Y-axis of each subplot.

    figsize: (opt) list or tuple
        Size of the figure to plot: width x height.
        default: (4,4)

    subplots: (opt) list or tuple
        Definition of subplots to include in the figure: height x width.
        default: (1,1)

    yref: (opt) str
        Name of the column in `df` containing a reference Y-value to plot.
        The value is taken from the first observation in `df`.
        default: None

    dict_options: (opt) dict
        Dictionary of plotting options.
        default: {  'axis': {   'fontsize': 13,
                                'limits': {'xmin': None, 'xmax': None,
                                           'ymin': 0, 'ymax': None,
                                           'allequal': True},    # This refers only to the Y-axis scale, not the X-axis
                                # Should the Y-axis be removed when NOT the leftmost Y-axis?
                                # (only makes sense when the 'allequal' attribute above is True,
                                # i.e. when all axis have the same scale)
                                'remove_yaxis': False,
                                # Name of variables to use as secondary tickmarks
                                'tickmarks': {'x2': None, 'y2': None}
                            },
                    'labels':   {'x': None, 'y': None, 'x2': None, 'y2': None},
                                    ## If different x and y variables are to be plotted,
                                    ## labels should be given as lists having the same length
                                    ## as the number of subplots to make,
                                    ## numbered sequentially along the rows first.
                    'multipliers':  {'x': 1, 'y': 1, 'error': 2},
                    'properties':   {# Properties for points for each subplot
                                     # (either one value or a list of values, one for each subplot)
                                    'color': "black", 'marker': '.', 'markersize': 5, 'linestyle': 'None',
                                    # Properties for error bars and centers
                                    # (either one value or a list of values, one for each subplot)
                                    'color_center': "blue", 'marker_center': "x", 'linestyle_center': "solid",
                                    'capsize': 4
                                    },
                    'showtitle':     True,
                    'stats':        {'center': "mean", 'scale': "std"},
                    'texts':        {'title': None}
                }

    Return: list of matplotlib.axes._subplots.AxesSubplot
    List of AxesSubplot objects containing each subplot generated by the function. 
    """
    def setup_properties(dict_properties, nplots):
        "Make sure there is one property value per plot when only one value is given"
        for key, value in dict_properties.items():
            if not isinstance(value, list):
                # Convert the value to a list of length 1 (e.g. ["red", "red"] for property 'color')
                dict_properties[key] = [value]*nplots
            elif len(value) != nplots:
                raise ValueError("The number of plot property values for key {} in the properties dictionary (value={}) must match the number of plots to produce ({})" \
                                 .format(key, value, nplots))

    #----- Parse input parameters -----
    # Parse options
    dict_options_default = default_plot_options()
    parse_dict_params(dict_options, dict_options_default)

    (nrows, ncols) = (subplots[0], subplots[1])
    n_subplots = nrows * ncols
    setup_properties(dict_options['properties'], n_subplots)

    # Convert any variable that is a string to a LIST that states that the same variable name is used for all plots
    if isinstance(xvars, str) and isinstance(yvars, str):
        warn("More than one subplot was requested but all 'x' and 'y' variables to plot are the same.")
    if isinstance(xvars, str):
        xvars = [xvars]*n_subplots
    if isinstance(yvars, str):
        yvars = [yvars]*n_subplots
    if isinstance(dict_options['labels']['x'], str):
        dict_options['labels']['x'] = [ dict_options['labels']['x'] ] * n_subplots 
    if isinstance(dict_options['labels']['y'], str):
        dict_options['labels']['y'] = [ dict_options['labels']['y'] ] * n_subplots
    if isinstance(dict_options['labels']['x2'], str):
        dict_options['labels']['x2'] = [ dict_options['labels']['x2'] ] * n_subplots 
    if isinstance(dict_options['labels']['y2'], str):
        dict_options['labels']['y2'] = [ dict_options['labels']['y2'] ] * n_subplots
    if isinstance(dict_options['axis']['tickmarks']['x2'], str):
        dict_options['axis']['tickmarks']['x2'] = [ dict_options['axis']['tickmarks']['x2'] ] * n_subplots
    if isinstance(dict_options['axis']['tickmarks']['y2'], str):
        dict_options['axis']['tickmarks']['y2'] = [ dict_options['axis']['tickmarks']['y2'] ] * n_subplots

    if n_subplots > 1 and dict_options['axis']['remove_yaxis'] and not dict_options['axis']['limits']['allequal']:
        warn("The non-leftmost Y-axis will be removed from the plots but the Y-axis may be different!", \
             "\nSet dict_options['axis']['limits']['allequal']=True if you want to remove Y-axis from non-leftmost plots.")

    # Assertions
    assert isinstance(xvars, list)
    assert isinstance(yvars, list)
    assert isinstance(xvars, list)
    assert isinstance(yvars, list)
    assert len(xvars) == n_subplots
    assert len(yvars) == n_subplots
    #----- Parse input parameters -----

    # Create the axes on which the plots will be produced
    axes = plt.figure(figsize=figsize).subplots(nrows=subplots[0], ncols=subplots[1])
    if n_subplots == 1:
        # Convert axes to an array, so that everything works fine from now on
        axes = np.array([axes])
    # Convert the axes array to a 1D array so that we can easily go over all of them
    axes1D = axes.reshape((n_subplots,), order='C')    # 'C' is the default, i.e. "C language"-order (along the rows first), as opposed to Fortran order (along the columns first)

    for k, ax in enumerate(axes1D):
        x = xvars[k]
        y = yvars[k]
        dict_options_k = copy.deepcopy(dict_options)
        for prop in dict_options['properties']:
            # Set the properties for the current plot from the k-th entry of 'properties' in the input dictionary
            dict_options_k['properties'][prop] = dict_options['properties'][prop][k]
        legend_objects, legend_texts = plotting_func(ax, df, x, y, dict_options_k)
        if yref is not None:
            line_ref = ax.hlines(df.iloc[0][yref] * dict_options['multipliers']['y'],
                                 ax.get_xlim()[0], ax.get_xlim()[1],
                                 #np.min(df[x] * dict_options['multipliers']['x']),
                                 #np.max(df[x] * dict_options['multipliers']['x']),
                                 color='gray', linestyles='dashed')
            legend_objects += [line_ref]

        #-- Legend
        if yref is not None:
            legend_texts += [yref_legend is None and "" or yref_legend]
        ax.legend(legend_objects, legend_texts)#, fontsize='x-small')

    #-- Axis limits
    if dict_options['axis']['limits']['allequal']:
        #xmin = np.Inf
        #xmax = -np.Inf
        ymin = np.Inf
        ymax = -np.Inf
        for ax in axes1D:
            # Axis limits
            #xmin = min(xmin, ax.get_xlim()[0]); xmax = max(xmax, ax.get_xlim()[1])
            ymin = min(ymin, ax.get_ylim()[0]); ymax = max(ymax, ax.get_ylim()[1])
    else:
        #xmin = xmax = None
        ymin = ymax = None

    # Update any of the axis limits if they are given by the user
    # and set the axis labels, if any given
    for k, ax in enumerate(axes1D):
        x = xvars[k]
        y = yvars[k]
        xmin = dict_options['axis']['limits'].get('xmin')
        xmax = dict_options['axis']['limits'].get('xmax')
        ymin = dict_options['axis']['limits'].get('ymin', ymin)
        ymax = dict_options['axis']['limits'].get('ymax', ymax)
        if xmin is None:
            xmin = ax.get_xlim()[0]
        if xmax is None:
            xmax = ax.get_xlim()[1]
        if ymin is None:
            ymin = ax.get_ylim()[0]
        if ymax is None:
            ymax = ax.get_ylim()[1]
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Axis labels
        if dict_options['labels']['x'] is None:
            xlabel = x
        else:
            xlabel = dict_options['labels']['x'][k]
        if dict_options['labels']['y'] is None:
            ylabel = y
        else:
            ylabel = dict_options['labels']['y'][k]
        ax.set_xlabel(xlabel, fontsize=dict_options['axis']['fontsize'])
        if k == 0 or not dict_options['axis']['remove_yaxis']:
            ax.set_ylabel(ylabel, fontsize=dict_options['axis']['fontsize'])

        # Secondary X-axis tickmarks?
        if dict_options['axis']['tickmarks']['x2'] is not None:
            x2 = dict_options['axis']['tickmarks']['x2'][k]
            x_values = np.unique(df[x] * dict_options['multipliers']['x'])
            x2_values = np.unique(df[x2])
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) # Set the same X-lims as the primary X-axis so that the tickmark values are aligned with the original X-values
            ax2.set_xticks(x_values)
            ax2.set_xticklabels(["{:,.0f}".format(x2) for x2 in x2_values], rotation=90)
            ax2.set_xlabel(dict_options['labels'].get('x2', "")[k])

        # Remove ticks and labels from the right plot as the axis is the same as on the left plot
        if dict_options['axis']['remove_yaxis']:
            if np.mod(k, ncols) > 0:
                # This is not the leftmost y axis, so remove it         
                ax.yaxis.set_ticks([]); ax.yaxis.set_ticklabels([])

    if dict_options['showtitle'] and dict_options['texts']['title'] is not None:
        plt.suptitle(dict_options['texts']['title'])

    return axes

def plot_errorbars(ax, df, x, y, dict_options):
    """
    Generates a points + line plot with error bars around each (x, yc) value
    where yc is the center of all y's measured for each x (e.g. their mean).

    Arguments:
    ax: matplotlib.axes._subplots.AxesSubplot
        AxesSubplots object where the plot should be generated, e.g. an instance of a subplot
        returned by the matplotlib.pyplot.figure().subplots() method.

    df: pandas Dataframe
        Dataframe containing the data to plot.

    x: str
        Name of the column in `df` to use on the X-axis.

    y: str
        Name of the column in `df` to use on the Y-axis.

    dict_options: dict
        Dictionary of options containing at least the following attributes:
        - multipliers: (defining a scalar by which each plotted variable is multiplied)
            -> x: multiplier for the X values
            -> y: multiplier for the Y values (e.g. 100 to show percentages)
            -> error: multiplier for the "standard error" associated to the bar height (e.g. 2)
        - properties: (defining the garph properties)
            -> color: color for each replication (e.g. "black")
            -> marker: marker for each replication (e.g. '.')
            -> markersize: size of the marker for each replication (e.g. 5)
            -> capsize: capsize parameter of the matplotlib.pyplot.errorbar() function defining the size of the cap of the error bars
            -> color_center: color of the marker for the center of the error bar (e.g. "red") 
            -> marker_center: marker for the center of the error bar (e.g. "x")
            -> linestyle_center: style for the line connecting the error bar centers (e.g. "solid")
        - stats:
            -> center: name of the function that defines the center of the error bars (e.g. "mean")
            -> scale: name of the function that defines the height of the error bars (e.g. "std");
                        the height is defined by `multiplier-error` * `scale`/sqrt(n) * 'multiplier-y',
                        where n is the number of replications at the corresponding `x` value.

    Return: tuple
    Duple containing:
    - the list of legend objects indicated by the legend texts
    - the list of legend texts 
    """
    #------- Parse input parameters
    assert isinstance(x, str)
    assert isinstance(y, str)
    dict_options_default = default_plot_options()
    parse_dict_params(dict_options, dict_options_default)
    #------- Parse input parameters

    #-- Plot the raw values
    points = ax.plot(df[x] * dict_options['multipliers']['x'],
                     df[y] * dict_options['multipliers']['y'],
                     color=dict_options['properties']['color'],
                     marker=dict_options['properties']['marker'], markersize=dict_options['properties']['markersize'],
                     linestyle='None')

    #-- Add error bars
    # Aggregate by x so that we can compute the center and error
    df_groupby_x = aggregation_bygroups(df, [x], [y],
                                        stats=['count', dict_options['stats']['center'], dict_options['stats']['scale']])

    errors = ax.errorbar(list(df_groupby_x.index * dict_options['multipliers']['x']),
                         df_groupby_x[ y ][ dict_options['stats']['center'] ] * dict_options['multipliers']['y'],
                         yerr=dict_options['multipliers']['error'] * df_groupby_x[ y ][ dict_options['stats']['scale'] ] / np.sqrt(df_groupby_x[ y ]['count']) * dict_options['multipliers']['y'],
                         capsize=dict_options['properties']['capsize'],
                         color=dict_options['properties']['color_center'],
                         marker=dict_options['properties']['marker_center'],
                         linestyle=dict_options['properties']['linestyle_center'])
    legend_objects = [points[0], errors]
    legend_texts = [y + (dict_options['multipliers']['y'] == 1 and " " or "*{}".format(dict_options['multipliers']['y'])),
                    "Avg(" + y + ") +/- " + (dict_options['multipliers']['error'] == 1 and " " or str(dict_options['multipliers']['error'])) + "SE"]

    return legend_objects, legend_texts

def plot_violins(ax, df, x, y, dict_options):
    """
    Generates a jittered-points + violin-plots for each `x` value which is assumed to contain
    replicated measures of `y`.
    Points are always shown in black color.

    Arguments:
    ax: matplotlib.axes._subplots.AxesSubplot
        AxesSubplots object where the plot should be generated, e.g. an instance of a subplot
        returned by the matplotlib.pyplot.figure().subplots() method.

    df: pandas Dataframe
        Dataframe containing the data to plot.

    x: str
        Name of the column in `df` to use on the X-axis.

    y: str
        Name of the column in `df` to use on the Y-axis.

    dict_options: dict
        Dictionary of options containing at least the following attributes:
        - multipliers: (defining a scalar by which each plotted variable is multiplied)
            -> x: multiplier for the X values
            -> y: multiplier for the Y values (e.g. 100 to show percentages)
        - properties: (defining the garph properties)
            -> color: color of the violins, their lines and their centers
            -> marker: marker for each replication (e.g. '.')
            -> markersize: size of the marker for each replication (e.g. 5)

    Return: tuple
    Duple containing:
    - the list of legend objects indicated by the legend texts
    - the list of legend texts 
    """
    #------- Parse input parameters
    assert isinstance(x, str)
    assert isinstance(y, str)
    dict_options_default = default_plot_options()
    parse_dict_params(dict_options, dict_options_default)
    #------- Parse input parameters

    fraction_violin_widths = 0.1
    x_values = np.unique(df[x])
    if len(x_values) > 1:
        violin_widths = fraction_violin_widths * (x_values[-1] - x_values[0]) * dict_options['multipliers']['x']
    else:
        violin_widths = fraction_violin_widths * x_values[0] * dict_options['multipliers']['x']
    col = dict_options['properties']['color']
    violinplot(ax,  [df[ df[x]==xvalue ][y] * dict_options['multipliers']['y'] for xvalue in x_values],
                    positions=x_values * dict_options['multipliers']['x'],
                    showmeans=True, showmedians=False, linewidth=2, widths=violin_widths,
                    color_body=col, color_lines=col, color_means=col)

    # Add the observed points
    npoints = df.shape[0]
    jitter = 1 + 0.1*(np.random.random(npoints) - 0.5)
    points = ax.plot(df[x] * dict_options['multipliers']['x'] * jitter,
                     df[y] * dict_options['multipliers']['y'],
                     color="black",
                     marker=dict_options['properties']['marker'], markersize=dict_options['properties']['markersize'],
                     linestyle='None')

    legend_objects = [points[0]]
    legend_texts = [y + (dict_options['multipliers']['y'] == 1 and " " or "*{}".format(dict_options['multipliers']['y']))]

    return legend_objects, legend_texts

def violinplot(axis, dataset, positions=None, showmeans=True, showmedians=True,
               color_body=None, color_lines=None, color_means=None, color_medians=None,
               widths=0.1, linewidth=2):
    """
    Create customized violin plots in the given axis generated with the matplotlib.pyplot module

    IMPORTANT: the matplotlib.pyplot.violinpot() function does NOT accept an empty list as part of the
    `dataset` for plotting and any NaN values present make the whole violin plot (e.g. for a given group)
    not be shown!   

    Arguments:
    axis: matplotlib.Axes
        Axis where the plot should be generated (possibly added to an existing plot).

    dataset:
        The `dataset` argument of the violinplot() function.
        A violin plot is generated for each separate element present in this argument.
        For instance, `dataset` could be a list of lists, in which case one separate violin plot
        is plotted for each list at the positions given in `positions`.

    positions:
        The `positions` argument of the violinplot() function.
        Normally a list of numbers of axis coordinates where the violin plots should be plotted.

    widths:
        The `widths` argument of the violinplot() function.
        This is a VERY IMPORTANT parameter as it defines the width of the violins themselves.
        Its value can be a scalar and it has to be in the scale of the horizontal axis, so that
        the violins themselves are visible!
        E.g. it could be defined as (max(positions) - min(positions)) / 10 

    Return:
    The customized output of the matplotlib.pyplot.violinplot() function, with colors changed.
    """
    violinplot_parts = axis.violinplot(dataset,
                                       positions=positions, showmeans=showmeans, showmedians=showmedians,
                                       widths=widths)
    # Set the color of body and lines
    # Ref:
    # - body: https://matplotlib.org/3.2.1/gallery/statistics/customized_violin.html)
    # - lines: https://pythonhealthcare.org/2018/04/13/52-matplotlib-violin-plots/
    if color_body is not None:
        for body in violinplot_parts['bodies']:
            body.set_facecolor(color_body)
            body.set_edgecolor(color_body)
            #body.set_alpha(0.3)

    if color_lines is not None:
        for lines in ('cbars','cmins','cmaxes'):
            line = violinplot_parts[lines]
            line.set_edgecolor(color_lines)
            line.set_linewidth(linewidth)
            
    if showmeans and color_means is not None:
        violinplot_parts['cmeans'].set_edgecolor(color_means)

    if showmedians:
        color_median = color_lines if color_medians is None else color_medians
        violinplot_parts['cmedians'].set_edgecolor(color_median)

    return violinplot_parts
