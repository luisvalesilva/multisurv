"""Utility methods for graph plotting."""


import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def show_default_colors():
    def despine(fig):
        # Hide spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Hide ticks and labels
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return ax

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_subplot(111)

    for i, color in enumerate(colors):
        xy = (i, 0)
        rect = matplotlib.patches.Rectangle(xy, 1, 1, color=color)
        ax.add_patch(rect)

    despine(ax)
    plt.xlim([0, i + 1])
    plt.ylim([0, 1])
    plt.show()

def save_plot_for_figure(figure, file_name, path=None):
    """Save matplotlib plot to pdf file.

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        Output of Model.evaluate().
    file_name: str
        Name for the output file.
    path: str
        Path to location to save file to.
    Returns
    -------
    Figure as pdf file.
    """
    file_extension = '.pdf'
    file_name += file_extension
    if path is not None:
        file_name = os.path.join(path, file_name)
    figure.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True)

def format_output_intervals(intervals):
    """Convert intervals to time points for plotting.

    Parameters
    ----------
    intervals: torch.Tensor
        Model output intervals in days. Can be obtained from the
        "output_intervals" attribute of a "Model" instance.
    Returns
    -------
    Prediction time points as numpy.ndarray.
    """ 
    # Convert to years
    time_points = np.array(intervals) / 365
    # Define center of each interval as time point
    time_points[1:] = time_points[1:] - np.diff(time_points)[0] / 2
                
    return time_points
