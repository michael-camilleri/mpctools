"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from mpctools.extensions.utils import dict_width
import numpy as np
import pandas


def DataFrame(data):
    """
    Wrapper around DataFrame constructor to support instantiation from a scalar

    :param data:
    :return:
    """
    try:
        return pandas.DataFrame(data)
    except ValueError:
        return pandas.DataFrame(data, index=[0,])


def build_dataframe(_d, _idx_names, _idx_values):
    """
    Creates a dataframe of arbitrary length with a specific index set

    :param _d:          Dictionary of Key-Value pairs or a Pandas DataFrame
    :param _idx_names:  Ordered list of Index Columns
    :param _idx_values: Ordered list of Index Values (correspond to columns)
    :return: Pandas DataFrame
    """
    # Find the length of each array in the dictionary and get maximum
    if type(_d) is dict:
        _w = dict_width(_d)
        _df = pandas.DataFrame(_d, index=np.arange(_w))
    else:
        _df = _d.reset_index(drop=True)

    # Create the Index Columns
    for c, idx in enumerate(_idx_names):
        _df[idx] = _idx_values[c]

    # Reset the Index variables
    _df.set_index(_idx_names, inplace=True)

    # Return DataFrame
    return _df


def recategorise(_df, _cat_type, _cols, _map=None):
    """
    A Convenience function to re-categorise the columns in the dataframe: the operation is performed in place. Note that
    the function allows mapping of categorical type with no overlaps through the _map construct.

    :param _df:         Data-Frame to recategorise
    :param _cat_type:   Categorical DType to set
    :param _cols:       The columns to change
    :param _map:        If need be, map certain values to others before recategorising: values which are not mapped
                        and which appear in the data but not in the new label-set (CDType) will be turned into NaN
    :return:            None (operation done in place)
    """
    if _map is not None:
        for col in _cols:
            _df.loc[:, col] = _df[col].map(
                _map
            )  # Note that this will automatically convert to the appropriate type!
            _df.loc[:, col] = _df[col].astype(float, copy=False).astype(_cat_type)
    else:
        for col in _cols:
            _df.loc[:, col] = _df[col].astype(float, copy=False).astype(_cat_type)


def nanmode(col):
    """
    Return the Mode of the column, including NaN as an indicator

    :param col: The col of data to work on
    :return:    The mode
    """
    return col.value_counts(dropna=False).idxmax()


def dfmultiindex(df, lvl, vals, indexer=False):
    """
    Provides a means of indexing a Dataframe based on an arbitrary index level and arbitrary index-values

    :param df:      The DataFrame to index
    :param lvl:     The Index level to act upon
    :param vals:    The Values to index upon
    :param indexer: If True, returns the indexer (slicer) instead of the actual data (allowing more flexible use)
    :return:        The chunk of DataFrame matching the index or the indexer
    """
    _slicer = tuple(vals if _l == lvl else slice(None) for _l in range(len(df.index.levels)))
    if indexer:
        return _slicer
    else:
        return df.loc[_slicer, :]


def time_overlap(tr1, tr2, time_only=False):
    """
    Method to check whether two time-ranges overlap.

    :param tr1: TimeRange 1 (tuple of pandas.DateTime)
    :param tr2: TimeRange 2 (tuple of pandas.DateTime)
    :param time_only: If true, then only consider overlap in absolute time-of-day rather than entire date.
    :return:    True if they overlap, false otherwise.
    """
    if time_only:
        return (tr1[0].time() <= tr2[1].time()) and (tr1[1].time() >= tr2[0].time())
    else:
        return (tr1[0] <= tr2[1]) and (tr1[1] >= tr2[0])


def segment_periods(df, fill_holes=False):
    """
    Segment the Dataframe Rows into periods where the rows are identical. Currently, identical is
      defined in terms of missing data.

    :param df: Dataframe, where time flows along the rows.
    :param fill_holes: If true, consider the first/last valid indices as the change-points.
    :return:  A Series with segment names.
    """
    if fill_holes:
        df = df.notnull()
        df[~df] = np.NaN
        for c in df.columns:
            df.loc[df[c].first_valid_index():df[c].last_valid_index(), c] = 1.0
    return df.notnull().diff(axis=0).any(axis=1).cumsum()
