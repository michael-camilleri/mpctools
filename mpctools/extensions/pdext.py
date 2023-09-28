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
import pandas as pd

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


def idx_where(df, cond, axis=0):
    """
    Return indices where condition is true

    :param df: DataFrame or series
    :param cond: Function to determine truth
    :param axis: For DataFrame: determines axis along which to operate
    :return:
    """
    if isinstance(df, pd.Series) or axis == 0:
        return [i for i in range(len(df)) if cond(df.iloc[i])]
    else:
        return [i for i in range(len(df.columns)) if cond(df.iloc[:, i])]


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


def argmax(df, axis=0, name='Max'):
    """
    Similar to idxmax but returns numeric index.

    ***Note***: Different from idxmax, it operates under the assumption that NaN's are skipped:
    however, rows [cols] which are all NaN are NaN.

    :param df:  DataFrame
    :param axis:  Axis along which to compute max
    :param name: Name for column
    :return: Numeric indices for other axis.
    """
    # Compute for which there is at least one non-null value
    _finite = df.dropna(axis=1-axis, how='all')
    _finite = pd.Series(
        np.nanargmax(_finite.to_numpy(), axis=axis),
        index=_finite.columns if axis == 0 else _finite.index,
        name=name
    )
    return _finite.reindex(df.index)


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


def diff(df, periods=1, fillna=False, axis=0):
    """
    Difference function which works with any datatype (not just integers)

    :param df: Pandas DataFrame or Series
    :param periods: How many shifts to compute for
    :param fillna: If True, then the NaNs from shifting are converted to 0
    :param axis: Axis along which to operate (for DataFrame only)
    :return: Difference along axis.
    """
    def __diff(s, p, n):
        d = s.ne(s.shift(periods=p).bfill()).astype(float)
        if not n:
            d.iloc[:p] = np.NaN
        return d

    if isinstance(df, pd.Series):
         return __diff(df, periods, fillna)
    else:
        return df.apply(axis=axis, func=__diff, p=periods, n=fillna)


def parallel_apply(df, func, axis=0, n_jobs=8, split=200, raw=False):
    """
    Apply a Function to entries of a Dataframe in parallel

    @param df: Dataframe to act on
    @param func: Function to execute. This will be applied per row/column and should in general return a Series, however
                 see also extend below.
    @param axis: Axis along which to apply: 0 to apply along the row (i.e. per column) and 1 vice-versa. if None, then
                 it is assumed that df is a groupby object that can be acted upon immediately. Note that for proper
                 grouping, the result of the application function should itself be a Dataframe (if a Series, convert to
                 Dataframe and Transpose)
    @param n_jobs: Number of jobs to run in parallel
    @param split:  The number of splits to do (when not grouped)
    @param raw: If True, then it is expected that the application of the function will return something other than a
                   dataframe, and can be handled by the user at the end. Only valid when not grouped
    @return: New Dataframe with applied
    """
    # Internal Functions
    @cl.contextmanager
    def progress(pbar):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        def tqdm_print_progress(self):
            pbar.update(value=self.n_completed_tasks)

        pbar.reset()
        original_print_progress = jl.parallel.Parallel.print_progress
        jl.parallel.Parallel.print_progress = tqdm_print_progress
        try:
            yield pbar
        finally:
            jl.parallel.Parallel.print_progress = original_print_progress

    def __parallel_apply(grp, _f, ax=0, _raw=False):
        if _raw:
            result = []
            for _, rc in (grp.items() if ax == 0 else grp.iterrows()):
                result.append(_f(rc))
            return result
        else:
            return grp.apply(_f, axis=ax)

    # Run in parallel
    if axis is not None:
        df = np.array_split(df, split)
        with progress(ProgressBar(len(df), prec=2)) as pbar:
            res = jl.Parallel(n_jobs=n_jobs, prefer='processes')(jl.delayed(__parallel_apply)(grp, func, axis, raw) for grp in df)
        return [r for rr in res for r in rr] if raw else pd.concat(res, axis=(1 - axis))
    else:
        with progress(ProgressBar(len(df), prec=2)) as pbar:
            res = jl.Parallel(n_jobs=n_jobs, prefer='processes')(jl.delayed(func)(grp) for _, grp in df)
        return pd.concat(res)
