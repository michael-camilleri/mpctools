"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from matplotlib.colors import LinearSegmentedColormap as lsc
from matplotlib import pyplot as plt, axes
from mpctools.extensions import npext
import seaborn as sns
import numpy as np
import warnings


def plot_matrix(matrix, mode='hinton', min_max=None, show_val=False, ax=None, cbar=None, labels=None, y_labels=None,
                x_rot=0, y_rot=0, fmt='.2f'):
    """
    Draw Hinton/Heatmap diagram for visualizing a weight matrix.

    Hinton diagrams are useful for visualizing the values of a 2D array (e.g. a weight matrix): Positive and negative
    values are represented by white and black squares, respectively, and the size of each square represents the
    magnitude of each value. Heatmaps on the other hand simply colour-code the weights.

    Rows (first-dimension) of the matrix are printed along the y-axis, with columns (2nd dimension) along the X-axis.
    Note that by default, the y-axes is inverted.

    Inspired by [Hinton Demo](https://matplotlib.org/examples/specialty_plots/hinton_demo.html). The heatmap makes
    use of seaborn functionality, with some pre/post processing.

    :param matrix:      2D Matrix to Display.
    :param mode:        Plotting Mode. Options are (not case sensitive):
                            'hinton' - Plot a Hinton diagram [Default]
                            'heatmap' - Plot a Heatmap
    :param min_max:     Range to plot at. This can be:
                            None: Infer from Data. For Hinton diagrams, this will infer one value, whereas it will
                                  infer separate minimum/maximum for Heatmaps
                            Float: Only valid for Hinton plots - use this as the maximum weight
                            Tuple/List: (minimum, maximum). Only for Heatmaps.
    :param show_val:    If True, then show the numerical value on the heatmap/hinton
    :param ax:          Axes to plot on
    :param cbar:        Only relevant when mode is 'heatmap': if not None, specifies a seperate axes for the colour bar.
                        If False, then do not plot a heatmap.
    :param labels:      Labels for the axes.
    :param y_labels:    If not None, then use separate labels for the y-axis.
    :param x_rot:       Rotation for the X-Axis Labels
    :param y_rot:       Rotation for the Y-Axis Labels
    :param fmt:         Formatting String for Value labels (if any)
    :return:
    """
    # Sort out the mode
    if mode.lower() == 'hinton':
        mode = True
    elif mode.lower() == 'heatmap':
        mode = False
    else:
        warnings.warn('Unrecognised Mode: Defaulting to Hinton plot', UserWarning)
        mode = True

    # Sort out the min_max:
    if mode:
        if min_max is None:
            min_max = np.power(2, np.ceil(np.log2(np.abs(matrix).max())))
        elif np.size(min_max) == 1:
            min_max = float(min_max)
        else:
            warnings.warn('Hinton Plot only accepts a single min_max value: inferring from data', UserWarning)
            min_max = np.power(2, np.ceil(np.log2(np.abs(matrix).max())))
    else:
        if min_max is None:
            min_max = [matrix.min(), matrix.max()]
        elif np.size(min_max) == 2:
            min_max = np.array(min_max, dtype=float)
        else:
            warnings.warn('Heatmap requires separate min_max values: inferring from data', UserWarning)
            min_max = [matrix.min(), matrix.max()]

    # Sort out axes
    ax = ax if ax is not None else plt.gca()

    # Plot
    if mode:
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        for (y, x), w in np.ndenumerate(matrix):
            size = np.sqrt(np.abs(w) / min_max)
            ax.add_patch(plt.Rectangle([x-size/2, y-size / 2], size, size, facecolor='white' if w > 0 else 'black',
                                       edgecolor='white' if w > 0 else 'black'))
            if show_val:
                ax.text(x, y, '{{:{}}}'.format(fmt).format(w), horizontalalignment='center', verticalalignment='center',
                        color='black' if w > 0 else 'white')
        ax.set_ylim(-1, matrix.shape[0])
        ax.set_xlim(-1, matrix.shape[1])
        ax.invert_yaxis()
    else:
        sns.heatmap(matrix, vmin=min_max[0], vmax=min_max[1], annot=show_val, fmt=fmt, ax=ax, cbar=cbar is not False,
                    cbar_ax=cbar)

    # Add Ticks/Labels
    if labels is not None:
        y_labels = labels if y_labels is None else y_labels
        if mode:
            ax.set_xticks(np.arange(len(labels)))
        else:
            ax.set_xticks(np.arange(0.5, len(labels) + 0.5))
        ax.set_xticklabels(labels, rotation=x_rot, horizontalalignment='center' if x_rot == 0 else 'right')
        if mode:
            ax.set_yticks(np.arange(len(y_labels)))
        else:
            ax.set_yticks(np.arange(0.5, len(y_labels) + 0.5))
        ax.set_yticklabels(y_labels, rotation=y_rot, verticalalignment='center' if y_rot == 0 else 'bottom')


def plot_blandaltman(series1, series2, mode, labels, model_names, fnt_size=15, ax=None, *args, **kwargs):
    """
    Generate a Bland-Altman Plot of the data in series1 and 2

    The plot allows handling of multiple classes. To support this, both series1 and series2 must be lists of numpy
    arrays. If the data is 1-dimensional, then just wrap in a list

    :param series1: The Lead Data: Difference will be Series 1 - Series 2
    :param series2: The other Data (in same order)
    :param mode:    The Bland-Altman diagram's abscissa can be in one of two forms: either the mean, or one of the data
                        series. If None, then the mean is plotted: else, if 0 plot series1, if 1 plot series2.
    :param labels:  Labels for the different classes
    :param model_names: The names of the Model
    :param fnt_size: Size of font for labels etc...
    :param ax:      Axes to plot on: if None, create own axes
    :param args:    Arguments to pass on to plt.scatter
    :param kwargs:  Key-Word Arguments to pass on to plt.scatter
    :return:
    """
    # Resolve Axes
    ax = ax if ax is not None else plt.gca()

    # Loop over data
    _g_diff = []    # Need to keep track of this
    _g_s1 = []
    _g_s2 = []
    for s1, s2, l in zip(series1, series2, labels):
        if mode is None:
            _x_axis = np.mean(np.asarray([s1, s2]), axis=0)
        else:
            _x_axis = np.asarray(s1) if mode == 0 else np.asarray(s2)
        _diff = s1 - s2
        _g_diff.extend(_diff)
        _g_s1.extend(s1)
        _g_s2.extend(s2)
        ax.scatter(_x_axis, _diff, label=l, *args, **kwargs)

    # Find globals:
    _md = np.mean(_g_diff)
    _sd = np.std(_g_diff)
    ax.axhline(_md, color='gray', linestyle='-')
    ax.axhline(_md + 1.96 * _sd, color='gray', linestyle='--')
    ax.axhline(_md - 1.96 * _sd, color='gray', linestyle='--')

    # Finally, add labels
    ax.set_xlabel('Sample Mean' if mode is None else model_names[mode], size=fnt_size)
    ax.set_ylabel('{0} - {1}'.format(*model_names), size=fnt_size)
    print('Percentage Positive: {0:.3f}%, Mean Diff: {1:.3f}, Mean S1: {2:.3f}, Mean S2: {3:.3f}, Length: {4}'
          .format(np.sum(np.asarray(_g_diff) >= 0) * 100.0 / len(_g_diff), _md, np.mean(_g_s1), np.mean(_g_s2), len(_g_diff)))


def plot_categorical(time_series, values, labels, nan=-1, cmap=None, ax=None, y_labels=None, cbar=None, fnt_size=15):
    """
    Plots categorical data in time as a colour-coded series

    The system has limited support for NaN values
    Ideas from:
       https://robinsones.github.io/Better-Plotting-in-Python-with-Seaborn/
       https://matplotlib.org/gallery/images_contours_and_fields/pcolor_demo.html

    :param time_series: Time-Series Data to plot. Must be a 2D array, with time along the columns.
    :param values:      The allowable values: in ascending order
    :param labels:      String description of categorical labels
    :param nan:         Value to replace NaN with
    :param cmap:        Color map to use. It must have at least as many distinct colours as num_labels
    :param ax:          An axis to plot: if not specified, uses the current axis.
    :param y_labels:    The labels for the y-axis
    :param cbar:        If True or an Axis object, then plots a colour bar (on the provided axis if any) - else nothing
    :param fnt_size:    Size to use for fonts
    :return:            Tuple consisting of Color plot Collections and optionally the color map
    """
    # Format Input
    (n_rows, n_cols) = time_series.shape
    N = len(labels)
    ax = plt.gca() if ax is None else ax
    cmap = 'tab20' if cmap is None else cmap
    y_labels = [str(l) for l in np.arange(0.5, n_rows, 1.0)] if y_labels is None else y_labels

    # Transform Data
    _data = []
    for row in range(n_rows):
        _data.append(time_series[row, :].copy())
        _data[row][np.isnan(_data[row])] = nan  # Convert NaN
        _data[row] = npext.value_map(_data[row], np.arange(N), _from=values)
    _data = np.vstack(_data)

    # Generate Discrete Colour Map
    base = plt.cm.get_cmap(cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    # Create and format colour plot
    plot = ax.pcolormesh(_data, cmap=lsc.from_list(cmap_name, color_list, N), vmin=-0.5, vmax=N-0.5)
    ax.set_ylim(-0.1, n_rows+0.1)
    for y in range(n_rows):
        ax.axhline(y+1.0, color='k')  # Horizontal separator line
    ax.set_yticks(np.arange(0.5, n_rows, 1.0))
    ax.set_yticklabels(y_labels, fontsize=fnt_size)
    ax.tick_params(labelsize=fnt_size)

    # Plot the Colour Bar if need be
    if cbar is True or isinstance(cbar, axes.Axes):
        colorbar = plt.colorbar(plot, ticks=np.arange(N), cax=cbar if isinstance(cbar, axes.Axes) else None)
        colorbar.ax.set_yticklabels(labels, fontsize=fnt_size)
    else:
        colorbar = None

    return plot, colorbar