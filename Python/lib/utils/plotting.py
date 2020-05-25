# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:07:56 2020

@author: Daniel Mastropietro
@description: Functions for customized plotting.
"""


def violinplots(axis, dataset, positions=None, showmeans=True, showmedians=True,
                color_body=None, color_lines=None, color_mean=None, color_median=None, linewidth=2):
    """
    Create customized violin plots in the given axis generated with the matplotlib.pyplot module
    
    @return The customized output of the matplotlib.pyplot.violinplot() function, with colors changed.
    """

    violinplot_parts = axis.violinplot(dataset,
                                      positions=positions, showmeans=showmeans, showmedians=showmedians)
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
            
    if showmeans and color_mean is not None:
        violinplot_parts['cmeans'].set_edgecolor(color_mean)

    if showmedians:
        color_median = color_lines if color_median is None else color_median
        violinplot_parts['cmedians'].set_edgecolor(color_median)

    return violinplot_parts