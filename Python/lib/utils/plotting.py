# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:07:56 2020

@author: Daniel Mastropietro
@description: Functions for customized plotting.
"""


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